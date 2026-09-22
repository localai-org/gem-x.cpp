#include "gemx.h"
#include "gemx_stream.h"
#include "live_detection.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
using session_ptr=std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)>;
using vitpose_ptr=std::unique_ptr<gemx_vitpose,decltype(&gemx_vitpose_destroy)>;
using yolox_ptr=std::unique_ptr<gemx_yolox,decltype(&gemx_yolox_destroy)>;
template<class T> void read(std::istream &in,T *value,size_t count=1){
    if(!in.read(reinterpret_cast<char *>(value),sizeof(T)*count))throw std::invalid_argument("truncated input");
}
template<class T> void write(std::ostream &out,const T *value,size_t count=1){
    if(!out.write(reinterpret_cast<const char *>(value),sizeof(T)*count))throw std::runtime_error("output write failed");
}
uint32_t number(const char *text){
    uint32_t value=0;auto end=text+std::strlen(text);auto result=std::from_chars(text,end,value);
    if(result.ec!=std::errc{}||result.ptr!=end)throw std::invalid_argument("invalid integer");
    return value;
}
float real(const char *text){
    char *end=nullptr;errno=0;float value=std::strtof(text,&end);
    if(errno||end==text||end!=text+std::strlen(text)||!std::isfinite(value))throw std::invalid_argument("invalid number");
    return value;
}
void api(gemx_status status,const char *message){if(status!=GEMX_OK)throw std::runtime_error(message);}

struct image_data {
    uint32_t width=0,height=0,stride=0;
    std::array<float,4> box{},camera{};
    std::vector<uint8_t> rgb;
};
image_data parse_image(std::istream &in){
    std::array<char,8> magic{};read(in,magic.data(),magic.size());
    if(std::string(magic.data(),magic.size())!="S3DIMG01")throw std::invalid_argument("wrong packed image magic");
    image_data value;read(in,&value.width);read(in,&value.height);read(in,&value.stride);
    read(in,value.box.data(),4);read(in,value.camera.data(),4);
    if(value.width<8||value.height<8||value.width>32766||value.height>32766||
       uint64_t(value.width)*value.height>16000000||value.stride!=value.width*3)
        throw std::invalid_argument("invalid packed image dimensions");
    value.rgb.resize(uint64_t(value.stride)*value.height);read(in,value.rgb.data(),value.rgb.size());
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing packed image data");
    return value;
}

image_data load_image(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open packed image");
    return parse_image(in);
}

std::array<float,1024> parse_pose_token(std::istream &in){
    std::array<char,8> magic{};read(in,magic.data(),8);
    if(std::string(magic.data(),8)!="S3DOUT01")throw std::invalid_argument("wrong body result magic");
    uint32_t tensors=0;read(in,&tensors);if(tensors<1||tensors>64)throw std::invalid_argument("invalid body tensor count");
    std::array<float,1024> result{};
    for(uint32_t tensor=0;tensor<tensors;++tensor){
        uint32_t name_size=0,type=0,rank=0;uint64_t elements=0;read(in,&name_size);
        if(name_size<1||name_size>64)throw std::invalid_argument("invalid tensor name");
        std::string name(name_size,'\0');read(in,name.data(),name.size());
        read(in,&type);read(in,&rank);read(in,&elements);
        if(rank>8||elements>(1ull<<28))throw std::invalid_argument("invalid tensor descriptor");
        uint64_t product=1;std::array<uint64_t,8> dimensions{};
        for(uint32_t i=0;i<rank;++i){read(in,&dimensions[i]);if(!dimensions[i]||dimensions[i]>(1ull<<28)/product)throw std::invalid_argument("invalid tensor shape");product*=dimensions[i];}
        if(product!=elements)throw std::invalid_argument("tensor element count mismatch");
        if(name=="pose_token"){
            if(type!=1||rank!=2||dimensions[0]!=1||dimensions[1]!=1024)
                throw std::invalid_argument("invalid GEM-X pose token");
            read(in,result.data(),result.size());return result;
        }else{
            in.seekg(static_cast<std::streamoff>(elements*4),std::ios::cur);
            if(!in)throw std::invalid_argument("truncated tensor payload");
        }
    }
    throw std::invalid_argument("body result lacks GEM-X pose token");
}

std::array<float,1024> load_pose_token(const std::string &path){
    for(unsigned attempt=0;attempt<600000;++attempt){
        std::error_code error;
        if(std::filesystem::file_size(path,error)>=400000&&!error)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("body result did not become available");
    return parse_pose_token(in);
}

struct pose_sample {
    std::array<float,77*3> positions{},camera_positions{},local_translations{},keypoints{};
    std::array<float,77*4> rotations{};
    std::array<int32_t,77> parents{};
    std::array<float,3> camera{};
};
void save_pose(const std::string &path,const pose_sample &value){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create GEM-X pose");
    write(out,"GEMPOSE2",8);write(out,value.positions.data(),value.positions.size());
    write(out,value.camera_positions.data(),value.camera_positions.size());
    write(out,value.rotations.data(),value.rotations.size());
    write(out,value.local_translations.data(),value.local_translations.size());
    write(out,value.parents.data(),value.parents.size());write(out,value.camera.data(),3);
    write(out,value.keypoints.data(),value.keypoints.size());
}

void save_predictions(const std::string &path,uint32_t frames,
                      const std::vector<float> &motion,const std::vector<float> &camera){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot create raw GEM-X predictions");
    write(out,"GEMRAW01",8);write(out,&frames);
    write(out,motion.data(),motion.size());write(out,camera.data(),camera.size());
}
pose_sample parse_pose(std::istream &in){
    std::array<char,8> magic{};read(in,magic.data(),8);
    const bool version_two=std::string(magic.data(),8)=="GEMPOSE2";
    if(!version_two&&std::string(magic.data(),8)!="GEMPOSE1")throw std::invalid_argument("wrong GEM-X pose magic");
    pose_sample value;read(in,value.positions.data(),value.positions.size());
    if(version_two)read(in,value.camera_positions.data(),value.camera_positions.size());
    read(in,value.rotations.data(),value.rotations.size());
    read(in,value.local_translations.data(),value.local_translations.size());
    read(in,value.parents.data(),value.parents.size());read(in,value.camera.data(),3);
    read(in,value.keypoints.data(),value.keypoints.size());
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing GEM-X pose data");
    return value;
}

pose_sample load_pose(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open GEM-X pose");
    return parse_pose(in);
}

gemx_session_config config(const char *model,const char *module,const char *backend,
                           const char *description,uint32_t device,uint32_t threads,uint32_t cache){
    return {model,module,backend,std::strcmp(description,"-")?description:"",device,threads,cache};
}

float iou(const std::array<float,4> &a,const std::array<float,4> &b){
    const float x0=std::max(a[0],b[0]),y0=std::max(a[1],b[1]);
    const float x1=std::min(a[2],b[2]),y1=std::min(a[3],b[3]);
    const float intersection=std::max(0.f,x1-x0)*std::max(0.f,y1-y0);
    const float aa=std::max(0.f,a[2]-a[0])*std::max(0.f,a[3]-a[1]);
    const float ab=std::max(0.f,b[2]-b[0])*std::max(0.f,b[3]-b[1]);
    return intersection/(aa+ab-intersection+1e-7f);
}

// The state, noise matrices and two-stage association match the compact
// ByteTrack implementation released by NVIDIA with GEM-X.
struct kalman_track {
    std::array<double,8> x{};std::array<double,64> P{};
    int id=0,hits=1,lost=0;float score=0;
    kalman_track(const std::array<float,4> &box,float confidence,int identity):id(identity),score(confidence){
        const double w=box[2]-box[0],h=box[3]-box[1];
        x={(box[0]+box[2])*.5,(box[1]+box[3])*.5,w/(h+1e-7),h,0,0,0,0};
        for(int i=0;i<8;++i)P[i*8+i]=i<4?10.:1000.;
    }
    std::array<float,4> box() const{
        const double h=x[3],w=x[2]*h;
        return {float(x[0]-w*.5),float(x[1]-h*.5),float(x[0]+w*.5),float(x[1]+h*.5)};
    }
    void predict(){
        for(int i=0;i<4;++i)x[i]+=x[i+4];
        auto old=P;
        for(int r=0;r<8;++r)for(int c=0;c<8;++c){
            double value=old[r*8+c];
            if(r<4)value+=old[(r+4)*8+c];
            if(c<4)value+=old[r*8+c+4];
            if(r<4&&c<4)value+=old[(r+4)*8+c+4];
            P[r*8+c]=value;
        }
        for(int i=0;i<8;++i)P[i*8+i]+=i<4?1.:.01;
        ++lost;
    }
    void update(const std::array<float,4> &box,float confidence){
        const double w=box[2]-box[0],h=box[3]-box[1];
        std::array<double,4> z{(box[0]+box[2])*.5,(box[1]+box[3])*.5,w/(h+1e-7),h};
        double augmented[4][8]{};
        for(int r=0;r<4;++r){
            for(int c=0;c<4;++c)augmented[r][c]=P[r*8+c]+(r==c?(r==2?10.:1.):0.);
            augmented[r][4+r]=1.;
        }
        for(int column=0;column<4;++column){
            int pivot=column;for(int row=column+1;row<4;++row)
                if(std::abs(augmented[row][column])>std::abs(augmented[pivot][column]))pivot=row;
            if(std::abs(augmented[pivot][column])<1e-12)throw std::runtime_error("singular ByteTrack covariance");
            if(pivot!=column)for(int c=0;c<8;++c)std::swap(augmented[pivot][c],augmented[column][c]);
            const double scale=augmented[column][column];for(double &v:augmented[column])v/=scale;
            for(int row=0;row<4;++row)if(row!=column){const double factor=augmented[row][column];
                for(int c=0;c<8;++c)augmented[row][c]-=factor*augmented[column][c];}
        }
        double K[8][4]{};
        for(int r=0;r<8;++r)for(int c=0;c<4;++c)for(int k=0;k<4;++k)
            K[r][c]+=P[r*8+k]*augmented[k][4+c];
        std::array<double,4> residual{};for(int i=0;i<4;++i)residual[i]=z[i]-x[i];
        for(int r=0;r<8;++r)for(int c=0;c<4;++c)x[r]+=K[r][c]*residual[c];
        auto old=P;for(int r=0;r<8;++r)for(int c=0;c<8;++c){
            double value=old[r*8+c];for(int k=0;k<4;++k)value-=K[r][k]*old[k*8+c];P[r*8+c]=value;
        }
        ++hits;lost=0;score=confidence;
    }
};

std::vector<std::pair<int,int>> hungarian(const std::vector<std::vector<double>> &cost){
    if(cost.empty()||cost[0].empty())return {};
    const int rows=int(cost.size()),columns=int(cost[0].size());const bool transpose=rows>columns;
    const int n=transpose?columns:rows,m=transpose?rows:columns;
    auto value=[&](int row,int column){return transpose?cost[column][row]:cost[row][column];};
    std::vector<double> u(n+1),v(m+1);std::vector<int> p(m+1),way(m+1);
    for(int i=1;i<=n;++i){
        p[0]=i;int j0=0;std::vector<double> minimum(m+1,std::numeric_limits<double>::infinity());
        std::vector<bool> used(m+1);
        do{
            used[j0]=true;const int i0=p[j0];double delta=std::numeric_limits<double>::infinity();int j1=0;
            for(int j=1;j<=m;++j)if(!used[j]){const double current=value(i0-1,j-1)-u[i0]-v[j];
                if(current<minimum[j]){minimum[j]=current;way[j]=j0;}if(minimum[j]<delta){delta=minimum[j];j1=j;}}
            for(int j=0;j<=m;++j)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else minimum[j]-=delta;
            j0=j1;
        }while(p[j0]!=0);
        do{const int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);
    }
    std::vector<std::pair<int,int>> result;
    for(int j=1;j<=m;++j)if(p[j])result.push_back(transpose?std::pair{j-1,p[j]-1}:std::pair{p[j]-1,j-1});
    return result;
}

struct byte_tracker {
    std::vector<kalman_track> tracks;int next_id=0;
    struct association {std::vector<std::pair<int,int>> matched;std::vector<int> tracks,detections;};
    struct snapshot {std::array<float,4> box;int id;float score;};
    association match(const std::vector<int> &track_indices,const std::vector<gemx_detection> &detections,
                      const std::vector<int> &detection_indices,float threshold){
        association result;result.tracks=track_indices;result.detections=detection_indices;
        if(track_indices.empty()||detection_indices.empty())return result;
        std::vector<std::vector<double>> cost(track_indices.size(),std::vector<double>(detection_indices.size()));
        for(size_t i=0;i<track_indices.size();++i)for(size_t j=0;j<detection_indices.size();++j)
            cost[i][j]=1.-iou(tracks[track_indices[i]].box(),{detections[detection_indices[j]].box[0],detections[detection_indices[j]].box[1],detections[detection_indices[j]].box[2],detections[detection_indices[j]].box[3]});
        std::vector<bool> used_track(track_indices.size()),used_detection(detection_indices.size());
        for(auto [row,column]:hungarian(cost))if(1.-cost[row][column]>=threshold){
            result.matched.emplace_back(track_indices[row],detection_indices[column]);used_track[row]=true;used_detection[column]=true;
        }
        result.tracks.clear();result.detections.clear();
        for(size_t i=0;i<track_indices.size();++i)if(!used_track[i])result.tracks.push_back(track_indices[i]);
        for(size_t i=0;i<detection_indices.size();++i)if(!used_detection[i])result.detections.push_back(detection_indices[i]);
        return result;
    }
    std::vector<snapshot> advance(const gemx_detection *values,uint32_t count){
        std::vector<gemx_detection> detections(values,values+count);for(auto &track:tracks)track.predict();
        std::vector<int> all_tracks(tracks.size()),high,low;std::iota(all_tracks.begin(),all_tracks.end(),0);
        for(uint32_t i=0;i<count;++i)if(values[i].score>=.5f)high.push_back(i);else if(values[i].score>.1f)low.push_back(i);
        auto first=match(all_tracks,detections,high,.3f);
        for(auto [track,detection]:first.matched)tracks[track].update({values[detection].box[0],values[detection].box[1],values[detection].box[2],values[detection].box[3]},values[detection].score);
        auto second=match(first.tracks,detections,low,.3f);
        for(auto [track,detection]:second.matched)tracks[track].update({values[detection].box[0],values[detection].box[1],values[detection].box[2],values[detection].box[3]},values[detection].score);
        for(int detection:first.detections)tracks.emplace_back(
            std::array<float,4>{values[detection].box[0],values[detection].box[1],values[detection].box[2],values[detection].box[3]},values[detection].score,next_id++);
        tracks.erase(std::remove_if(tracks.begin(),tracks.end(),[](const auto &track){return track.lost>30;}),tracks.end());
        std::vector<snapshot> visible;for(const auto &track:tracks)if(track.lost==0)
            visible.push_back({track.box(),track.id,track.score});
        return visible;
    }
};

std::string read_string(std::istream &in){
    uint32_t size=0;read(in,&size);if(!size||size>4096)throw std::invalid_argument("invalid manifest path");
    std::string value(size,'\0');read(in,value.data(),size);
    if(value.find('\0')!=std::string::npos)throw std::invalid_argument("invalid manifest path data");
    return value;
}

std::vector<std::string> parse_paths(std::istream &in,const std::string &expected,uint32_t limit){
    std::array<char,8> magic{};read(in,magic.data(),8);
    if(std::string(magic.data(),8)!=expected)throw std::invalid_argument("wrong image manifest magic");
    uint32_t count=0;read(in,&count);if(!count||count>limit)throw std::invalid_argument("image manifest must contain 1..120 frames");
    std::vector<std::string> result;result.reserve(count);
    for(uint32_t i=0;i<count;++i)result.push_back(read_string(in));
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing image manifest data");
    return result;
}

std::vector<std::string> image_manifest(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open image manifest");
    return parse_paths(in,"GEMIMGS1",120);
}
struct sequence_manifest { bool explicit_boxes; uint32_t count; std::vector<std::string> images,bodies; std::vector<float> source_boxes; };
sequence_manifest parse_sequence(std::istream &manifest){
    std::array<char,8> magic{};read(manifest,magic.data(),8);
    const bool explicit_boxes=std::string(magic.data(),8)=="GEMSEQ02";
    if(!explicit_boxes && std::string(magic.data(),8)!="GEMSEQ01")throw std::invalid_argument("wrong sequence manifest magic");
    uint32_t count=0;read(manifest,&count);
    if(!count||count>120)throw std::invalid_argument("sequence manifest must contain 1..120 frames");
    std::vector<std::string> images(count),bodies(count);
    for(uint32_t i=0;i<count;++i){images[i]=read_string(manifest);bodies[i]=read_string(manifest);}
    std::vector<float> source_boxes(uint64_t(count)*3);
    if(explicit_boxes)read(manifest,source_boxes.data(),source_boxes.size());
    if(manifest.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing sequence manifest data");
    for(float x:source_boxes)if(!std::isfinite(x))throw std::invalid_argument("nonfinite manifest box");
    return {explicit_boxes,count,std::move(images),std::move(bodies),std::move(source_boxes)};
}

int detect_sequence(int argc,char **argv){
    if(argc!=10)throw std::invalid_argument("usage: gemx-pipeline --detect YOLOX MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS IMAGE_MANIFEST BOXES");
    char error[512]{};auto cfg=config(argv[2],argv[3],argv[4],argv[6],number(argv[5]),number(argv[7]),1);
    gemx_yolox *raw=nullptr;api(gemx_yolox_create(&cfg,&raw,error,sizeof(error)),error);
    yolox_ptr detector(raw,gemx_yolox_destroy);auto paths=image_manifest(argv[8]);byte_tracker tracker;
    std::vector<std::vector<byte_tracker::snapshot>> tracks;tracks.reserve(paths.size());
    std::vector<image_data> images;images.reserve(paths.size());std::unordered_map<int,double> area;
    for(const auto &path:paths){
        images.push_back(load_image(path));const auto &image=images.back();
        gemx_rgb_frame frame{image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,{}};
        std::array<gemx_detection,100> detections{};uint32_t count=0;
        api(gemx_yolox_detect(detector.get(),&frame,.5f,.45f,detections.data(),detections.size(),&count,error,sizeof(error)),error);
        tracks.push_back(tracker.advance(detections.data(),count));
        for(const auto &item:tracks.back())area[item.id]+=std::max(0.f,item.box[2]-item.box[0])*std::max(0.f,item.box[3]-item.box[1]);
    }
    int primary=-1;double greatest=-1;for(const auto &[id,value]:area)if(value>greatest){greatest=value;primary=id;}
    std::vector<std::array<float,4>> boxes(paths.size());std::vector<bool> valid(paths.size());
    for(size_t i=0;i<tracks.size();++i)for(const auto &item:tracks[i])if(item.id==primary){boxes[i]=item.box;valid[i]=true;break;}
    if(primary<0){for(size_t i=0;i<boxes.size();++i)boxes[i]={0,0,float(images[i].width-1),float(images[i].height-1)};}
    else{
        size_t first=0;while(first<valid.size()&&!valid[first])++first;
        for(size_t i=0;i<first;++i)boxes[i]=boxes[first];size_t previous=first;
        for(size_t i=first+1;i<valid.size();++i)if(valid[i]){
            for(size_t j=previous+1;j<i;++j){const float alpha=float(j-previous)/float(i-previous);
                for(int axis=0;axis<4;++axis)boxes[j][axis]=boxes[previous][axis]*(1-alpha)+boxes[i][axis]*alpha;}
            previous=i;
        }
        for(size_t i=previous+1;i<boxes.size();++i)boxes[i]=boxes[previous];
    }
    for(size_t i=0;i<boxes.size();++i)for(int axis=0;axis<4;++axis)
        boxes[i][axis]=std::clamp(boxes[i][axis],0.f,axis%2?float(images[i].height-1):float(images[i].width-1));
    auto unsmoothed=boxes;
    for(size_t i=0;i<boxes.size();++i)for(int axis=0;axis<4;++axis){
        boxes[i][axis]=0;for(int offset=-2;offset<=2;++offset){
            const size_t index=static_cast<size_t>(std::clamp<int64_t>(int64_t(i)+offset,0,int64_t(boxes.size()-1)));
            boxes[i][axis]+=unsmoothed[index][axis]/5.f;
        }
        const float limit=axis%2?float(images[i].height-1):float(images[i].width-1);
        boxes[i][axis]=std::clamp(boxes[i][axis],0.f,limit);
    }
    std::ofstream out(argv[9],std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create box output");
    write(out,"GEMBOX01",8);uint32_t count=boxes.size();write(out,&count);
    for(const auto &box:boxes)write(out,box.data(),4);
    return 0;
}

int offline_sequence(int argc,char **argv){
    if(argc!=12)throw std::invalid_argument("usage: gemx-pipeline --offline DENOISER VITPOSE MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS SEQUENCE_MANIFEST OUTPUT_DIR FPS");
    char error[512]{};
    std::ifstream manifest(argv[9],std::ios::binary);if(!manifest)throw std::runtime_error("cannot open sequence manifest");
    auto [explicit_boxes,count,images,bodies,source_boxes]=parse_sequence(manifest);
    auto den_cfg=config(argv[2],argv[4],argv[5],argv[7],number(argv[6]),number(argv[8]),2);
    gemx_session *raw_session=nullptr;api(gemx_session_create(&den_cfg,&raw_session,error,sizeof(error)),error);
    session_ptr session(raw_session,gemx_session_destroy);
    auto pose_cfg=config(argv[3],argv[4],argv[5],argv[7],number(argv[6]),number(argv[8]),4);
    gemx_vitpose *raw_pose=nullptr;api(gemx_vitpose_create(&pose_cfg,&raw_pose,error,sizeof(error)),error);
    vitpose_ptr pose(raw_pose,gemx_vitpose_destroy);
    std::vector<float> keypoints(uint64_t(count)*77*3),boxes(uint64_t(count)*3),K(uint64_t(count)*9),features(uint64_t(count)*1024),angular(uint64_t(count)*6);
    // Static camera motion is the identity rotation in the published 6D
    // representation. Six zeroes do not encode identity: Gram-Schmidt cannot
    // normalize either basis vector, and the invalid condition also poisons
    // the world-orientation rollout.
    for(uint32_t i=0;i<count;++i){angular[uint64_t(i)*6]=1.f;angular[uint64_t(i)*6+4]=1.f;}
    std::vector<image_data> loaded;loaded.reserve(count);
    for(uint32_t i=0;i<count;++i){
        loaded.push_back(load_image(images[i]));const auto &image=loaded.back();
        const float cx=(image.box[0]+image.box[2])*.5f,cy=(image.box[1]+image.box[3])*.5f,size=image.box[2]-image.box[0];
        boxes[uint64_t(i)*3]=cx;boxes[uint64_t(i)*3+1]=cy;boxes[uint64_t(i)*3+2]=size;
        // GEM/ViTPose use the original cx/cy/size. Reconstructing them from
        // Body's float32 corner box loses bits and can change crop pixels.
        if(explicit_boxes)std::copy_n(source_boxes.data()+uint64_t(i)*3,3,boxes.data()+uint64_t(i)*3);
        const float focal=static_cast<float>(std::max(image.width,image.height));
        std::array<float,9> matrix={focal,0,image.width*.5f,0,focal,image.height*.5f,0,0,1};
        std::copy(matrix.begin(),matrix.end(),K.begin()+uint64_t(i)*9);
        auto token=load_pose_token(bodies[i]);std::copy(token.begin(),token.end(),features.begin()+uint64_t(i)*1024);
    }
    for(uint32_t start=0;start<count;start+=4){const uint32_t batch=std::min(4u,count-start);
        std::vector<gemx_rgb_frame> frames(batch);for(uint32_t i=0;i<batch;++i){const auto &image=loaded[start+i];
            frames[i]={image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,
                {boxes[uint64_t(start+i)*3],boxes[uint64_t(start+i)*3+1],boxes[uint64_t(start+i)*3+2]}};}
        api(gemx_vitpose_infer_rgb(pose.get(),frames.data(),batch,keypoints.data()+uint64_t(start)*77*3,
                                   uint64_t(batch)*77*3,error,sizeof(error)),error);
    }
    gemx_sequence_view input{count,keypoints.data(),boxes.data(),K.data(),features.data(),angular.data()};
    std::vector<float> raw_motion(uint64_t(count)*585),raw_camera(uint64_t(count)*3);
    const bool contact=std::strcmp(argv[1],"--offline-contact")==0;
    std::vector<float> contact_logits(uint64_t(count)*6);
    if(contact)api(gemx_infer_contacts(session.get(),&input,raw_motion.data(),raw_motion.size(),raw_camera.data(),raw_camera.size(),
        contact_logits.data(),contact_logits.size(),error,sizeof(error)),error);
    else api(gemx_infer(session.get(),&input,raw_motion.data(),raw_motion.size(),raw_camera.data(),raw_camera.size(),error,sizeof(error)),error);
    std::vector<float> body(uint64_t(count)*76*3),identity(uint64_t(count)*45),scales(uint64_t(count)*69),
        orient_camera(uint64_t(count)*3),translation_camera(uint64_t(count)*3),orient_world(uint64_t(count)*3),translation_world(uint64_t(count)*3);
    gemx_motion_view motion{count,body.data(),identity.data(),scales.data(),orient_camera.data(),translation_camera.data(),orient_world.data(),translation_world.data()};
    api(gemx_decode_predictions(session.get(),&input,raw_motion.data(),raw_motion.size(),raw_camera.data(),raw_camera.size(),&motion,error,sizeof(error)),error);
    if(contact)api(gemx_refine_contacts(session.get(),&motion,contact_logits.data(),contact_logits.size(),error,sizeof(error)),error);
    std::vector<float> positions(uint64_t(count)*77*3),rotations(uint64_t(count)*77*4),translations(uint64_t(count)*77*3);std::array<int32_t,77> parents{};
    gemx_skeleton_view skeleton{count,positions.data(),rotations.data(),parents.data(),translations.data()};
    api(gemx_build_skeleton(session.get(),&motion,&skeleton,error,sizeof(error)),error);
    // The exported animation uses world-space trajectory, while a source-image
    // overlay needs the articulated body under the checkpoint's camera
    // orientation and translation. Build that second position stream once.
    std::vector<float> camera_positions(uint64_t(count)*77*3),camera_root(uint64_t(count)*3);
    std::vector<float> camera_rotations(uint64_t(count)*77*4),camera_translations(uint64_t(count)*77*3);
    std::array<int32_t,77> camera_parents{};
    gemx_motion_view camera_motion{count,body.data(),identity.data(),scales.data(),orient_camera.data(),
        translation_camera.data(),orient_camera.data(),camera_root.data()};
    gemx_skeleton_view camera_skeleton{count,camera_positions.data(),camera_rotations.data(),
        camera_parents.data(),camera_translations.data()};
    api(gemx_build_skeleton(session.get(),&camera_motion,&camera_skeleton,error,sizeof(error)),error);
    if(camera_parents!=parents)throw std::runtime_error("camera and world skeleton topology differ");
    std::error_code filesystem_error;std::filesystem::create_directories(argv[10],filesystem_error);
    if(filesystem_error)throw std::runtime_error("cannot create sequence output directory");
    save_predictions((std::filesystem::path(argv[10])/"predictions.bin").string(),count,raw_motion,raw_camera);
    for(uint32_t i=0;i<count;++i){pose_sample sample;
        std::copy_n(positions.data()+uint64_t(i)*77*3,77*3,sample.positions.data());
        std::copy_n(camera_positions.data()+uint64_t(i)*77*3,77*3,sample.camera_positions.data());
        std::copy_n(rotations.data()+uint64_t(i)*77*4,77*4,sample.rotations.data());
        std::copy_n(translations.data()+uint64_t(i)*77*3,77*3,sample.local_translations.data());sample.parents=parents;
        std::copy_n(translation_camera.data()+uint64_t(i)*3,3,sample.camera.data());std::copy_n(keypoints.data()+uint64_t(i)*77*3,77*3,sample.keypoints.data());
        char name[32];std::snprintf(name,sizeof(name),"%06u.gpose",i);save_pose((std::filesystem::path(argv[10])/name).string(),sample);
    }
    const auto glb=std::filesystem::path(argv[10])/"motion.glb";
    api(gemx_export_skeleton_samples_glb(session.get(),&skeleton,real(argv[11]),glb.c_str(),error,sizeof(error)),error);
    return 0;
}

// Resident camera worker. Only observations are buffered; RGB frames and pose
// files are overwritten, so an arbitrarily long live session stays bounded.
int live_worker(int argc,char **argv){
    if(argc!=12 && argc!=13)throw std::invalid_argument("usage: gemx-pipeline --live-worker DENOISER VITPOSE YOLOX MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS WINDOW DIRECTORY [DETECT_INTERVAL]");
    char error[512]{};gemx_live *raw=nullptr;
    // The legacy worker respects its host's precision environment. The native
    // API also offers explicit strict-F32 validation for direct clients.
    api(gemx_live_create(argv[2],argv[3],argv[4],argv[5],argv[6],number(argv[7]),
        std::strcmp(argv[8],"-")?argv[8]:nullptr,number(argv[9]),number(argv[10]),argc==13?number(argv[12]):1,
        GEMX_LIVE_PARITY,GEMX_LIVE_BACKEND_DEFAULT,GEMX_LIVE_FRAME_INDEX,2000000,&raw,error,sizeof(error)),error);
    std::unique_ptr<gemx_live,decltype(&gemx_live_destroy)> pipeline(raw,gemx_live_destroy);
    std::array<int32_t,77> parents;api(gemx_live_topology(raw,parents.data(),77,error,sizeof(error)),error);
    const auto dir=std::filesystem::path(argv[11]);std::cout<<"READY"<<std::endl;
    std::ofstream profile;if(const char*path=std::getenv("GEMX_LIVE_PROFILE")){
        profile.open(path);if(!profile)throw std::runtime_error("cannot open live profile");
        profile<<"frame,context,read_ms,detector_ms,vitpose_ms,gem_decode_ms,world_skeleton_ms,camera_skeleton_ms,write_ms,total_ms,detector_ran\n";
    }
    const auto origin=std::chrono::steady_clock::now();uint64_t index=0;int64_t previous=-1;
    std::string command;
    while(std::getline(std::cin,command)){
        if(command=="RESET"){api(gemx_live_reset(raw,error,sizeof(error)),error);previous=-1;std::cout<<"RESET"<<std::endl;continue;}
        uint64_t seq=index;int64_t timestamp=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-origin).count();
        if(command!="FRAME"){
            std::istringstream in(command);std::string word,sequence_text,extra;in>>word>>sequence_text>>timestamp;
            auto parsed=std::from_chars(sequence_text.data(),sequence_text.data()+sequence_text.size(),seq);
            if(!in||word!="FRAME"||parsed.ec!=std::errc{}||parsed.ptr!=sequence_text.data()+sequence_text.size()||(in>>extra))throw std::invalid_argument("invalid live frame command");
        }else{
            if(previous==std::numeric_limits<int64_t>::max())throw std::invalid_argument("source clock exhausted; reset required");
            timestamp=std::max(timestamp,previous+1);
        } // legacy file ingestion time
        const auto start=std::chrono::steady_clock::now();auto image=load_image((dir/"frame.input").string());
        const auto read_end=std::chrono::steady_clock::now();gemx_live_result*result=nullptr;
        api(gemx_live_submit(raw,image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,seq,timestamp,nullptr,0,0,&result,error,sizeof(error)),error);
        std::unique_ptr<gemx_live_result,decltype(&gemx_live_result_destroy)> owned(result,gemx_live_result_destroy);
        uint64_t sequence,epoch,track;int64_t source_time;uint32_t status,flags;
        api(gemx_live_result_info(result,&sequence,&source_time,&epoch,&track,&status,&flags,error,sizeof(error)),error);
        auto copy=[&](uint32_t channel,auto&output){uint64_t count=0;api(gemx_live_result_copy(result,channel,output.data(),output.size(),&count,error,sizeof(error)),error);if(count!=output.size())throw std::runtime_error("live channel count");};
        std::array<float,9> metrics;copy(GEMX_LIVE_METRICS,metrics);
        const auto write_start=std::chrono::steady_clock::now();
        if(status==GEMX_LIVE_POSE){pose_sample sample;sample.parents=parents;
            copy(GEMX_LIVE_POSITIONS,sample.positions);copy(GEMX_LIVE_ROTATIONS,sample.rotations);copy(GEMX_LIVE_TRANSLATIONS,sample.local_translations);
            copy(GEMX_LIVE_CAMERA_POSITIONS,sample.camera_positions);copy(GEMX_LIVE_CAMERA_TRANSLATION,sample.camera);copy(GEMX_LIVE_KEYPOINTS,sample.keypoints);
            save_pose((dir/"pose.gpose").string(),sample);
        }
        auto elapsed=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        const auto end=std::chrono::steady_clock::now();
        if(profile){profile<<index<<','<<metrics[7]<<','<<elapsed(start,read_end);for(int i=0;i<5;++i)profile<<','<<metrics[i];profile<<','<<elapsed(write_start,end)<<','<<elapsed(start,end)<<','<<bool(flags&GEMX_LIVE_DETECTOR_RAN)<<'\n';}
        previous=source_time;++index;std::cout<<(status==GEMX_LIVE_POSE?"POSE ":"WARMUP ")<<uint32_t(metrics[6])<<std::endl;
    }
    return 0;
}

int export_sequence(int argc,char **argv){
    if(argc!=11)throw std::invalid_argument("usage: gemx-pipeline --export DENOISER MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS FPS MANIFEST OUTPUT");
    char error[512]{};auto cfg=config(argv[2],argv[3],argv[4],argv[6],number(argv[5]),number(argv[7]),1);
    gemx_session *raw=nullptr;api(gemx_session_create(&cfg,&raw,error,sizeof(error)),error);
    session_ptr session(raw,gemx_session_destroy);
    std::ifstream manifest(argv[9],std::ios::binary);if(!manifest)throw std::runtime_error("cannot open manifest");
    const auto paths=parse_paths(manifest,"GEMMAN01",GEMX_MAX_FRAMES);
    const uint32_t count=paths.size();
    std::vector<float> positions(uint64_t(count)*77*3),rotations(uint64_t(count)*77*4),translations(uint64_t(count)*77*3);
    std::array<int32_t,77> parents{};
    for(uint32_t i=0;i<count;++i){
        auto sample=load_pose(paths[i]);
        std::copy(sample.positions.begin(),sample.positions.end(),positions.begin()+uint64_t(i)*77*3);
        std::copy(sample.rotations.begin(),sample.rotations.end(),rotations.begin()+uint64_t(i)*77*4);
        std::copy(sample.local_translations.begin(),sample.local_translations.end(),translations.begin()+uint64_t(i)*77*3);
        if(i==0)parents=sample.parents;else if(parents!=sample.parents)throw std::invalid_argument("SOMA topology changed");
    }
    gemx_skeleton_view skeleton{count,positions.data(),rotations.data(),parents.data(),translations.data()};
    api(gemx_export_skeleton_samples_glb(session.get(),&skeleton,real(argv[8]),argv[10],error,sizeof(error)),error);
    return 0;
}
}

int main(int argc,char **argv){
    try{
        if(argc>1&&!std::strcmp(argv[1],"--live-worker"))return live_worker(argc,argv);
        if(argc>1&&!std::strcmp(argv[1],"--detect"))return detect_sequence(argc,argv);
        if(argc>1&&(!std::strcmp(argv[1],"--offline")||!std::strcmp(argv[1],"--offline-contact")))return offline_sequence(argc,argv);
        if(argc>1&&!std::strcmp(argv[1],"--export"))return export_sequence(argc,argv);
        throw std::invalid_argument("expected --detect, --offline, or --export");
    }catch(const std::exception &error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
