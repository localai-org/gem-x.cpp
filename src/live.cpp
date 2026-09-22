#include "gemx_stream.h"
#include "live_state.hpp"
#include "session.hpp"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <locale>
#include <mutex>
#include <sstream>

namespace {
using clock_type=std::chrono::steady_clock;
using gemx::require;
void call(gemx_status s,const char*err){if(s!=GEMX_OK)throw std::runtime_error(err);}
struct observation {std::array<float,231> keypoints{};std::array<float,3> box{};};
std::string quote(const std::string&s){std::string r="\"";for(unsigned char c:s){if(c=='"'||c=='\\'){r+='\\';r+=char(c);}else if(c<32){const char*hex="0123456789abcdef";r+="\\u00";r+=hex[c>>4];r+=hex[c&15];}else r+=char(c);}return r+'"';}
template<class T> void array(std::ostream&o,const T&v){o<<'[';bool first=true;for(auto x:v){if(!first)o<<',';first=false;o<<x;}o<<']';}
void names(std::ostream&o,const std::vector<std::string>&v){o<<'[';for(size_t i=0;i<v.size();++i){if(i)o<<',';o<<quote(v[i]);}o<<']';}
}
struct gemx_live_result {
 uint64_t sequence=0,epoch=0,track=0;int64_t timestamp=0;uint32_t outcome=GEMX_LIVE_WARMUP,flags=0;
 std::array<std::vector<float>,14> data;
};
struct gemx_live {
 std::mutex mutex;
 std::unique_ptr<gemx::session> session;
 std::unique_ptr<gemx_vitpose,decltype(&gemx_vitpose_destroy)> pose{nullptr,gemx_vitpose_destroy};
 std::unique_ptr<gemx_yolox,decltype(&gemx_yolox_destroy)> detector{nullptr,gemx_yolox_destroy};
 gemx::live_state state;
 std::deque<observation> history;
 uint32_t window=30,cadence=1,selection=0,precision=0,threads=8;
 int64_t gap=2000000;
 std::string definition,backend,module,gem_model,pose_model,detector_model;
 uint32_t device=0;
 std::array<int32_t,77> parents{};
};
namespace {
std::string definition(gemx_live&p){
 std::array<float,228> body{};std::array<float,45> identity{};std::array<float,69> scale{};scale[0]=1;
 std::array<float,3> zero{};
 gemx_motion_view motion{1,body.data(),identity.data(),scale.data(),zero.data(),zero.data(),zero.data(),zero.data()};
 auto rest=p.session->skeleton(motion);
 std::copy(rest.parents.begin(),rest.parents.end(),p.parents.begin());
 for(size_t i=0;i<24;++i)require(rest.names[gemx::smpl_mapping[i]]==gemx::soma_mapping_names[i],"unsupported SOMA joint mapping");
 std::array<float,72> smpl{};std::array<float,4> anchor{};char err[256];
 call(gemx_soma_to_smpl(rest.positions.data(),231,zero.data(),3,smpl.data(),72,anchor.data(),4,err,sizeof(err)),err);
 const std::array<int32_t,24> smpl_parents{-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21};
 std::array<float,72> smpl_rest_local=smpl;std::array<float,96> smpl_rest_rotations{};
 for(size_t i=0;i<24;++i){smpl_rest_rotations[i*4]=1;for(size_t k=0;k<3;++k)if(smpl_parents[i]>=0)smpl_rest_local[i*3+k]-=smpl[smpl_parents[i]*3+k];}
 std::ostringstream o;o.imbue(std::locale::classic());o<<std::setprecision(9);
 o<<"{\"schema\":\"gemx.motion.definition.v1\",\"mode\":\"live-absent-image\",\"config\":{\"window\":"<<p.window<<",\"detector_interval\":"<<p.cadence
  <<",\"selection\":"<<quote(p.selection==0?"upstream-largest-fallback":"unique-iou-0.3")<<",\"precision\":"<<quote(p.precision==0?"strict-f32":"backend-default")
  <<",\"backend\":"<<quote(p.backend)<<",\"device_index\":"<<p.device<<",\"backend_module\":"<<quote(p.module)
  <<",\"model_paths\":{\"gem\":"<<quote(p.gem_model)<<",\"vitpose\":"<<quote(p.pose_model)<<",\"yolox\":"<<quote(p.detector_model)<<"}"
  <<",\"graph_cache_capacity\":"<<std::min(p.window,32u)<<",\"threads\":"<<p.threads<<",\"max_gap_us\":"<<p.gap<<",\"device\":"<<quote(p.session->device())<<"},"
  "\"capabilities\":{\"body_required\":false,\"mesh\":false,\"contacts\":false,\"pose_confidence_3d\":false,\"keypoint_confidence_2d\":true,\"min_warmup_frames\":2,\"smoothing\":false,\"shared_weights\":false,\"queue_capacity\":0,\"interruptible\":false},"
  "\"timing\":{\"unit\":\"source_microseconds\",\"origin\":\"caller-declared\",\"model_policy\":\"accepted-frame-index\",\"continuous_world_trajectory\":false},"
  "\"soma77\":{\"schema\":\"gemx.soma77.v1\",\"joint_count\":77,\"root\":0,\"units\":\"metres\",\"handedness\":\"right\",\"basis\":\"native SOMA Y-up; no display conversion\",\"space\":\"gravity-aligned with pelvis translation zero, root rotation retained\",\"quaternion_order\":\"xyzw\",\"quaternion_policy\":\"unit, canonical w>=0, no temporal sign smoothing\",\"shape_policy\":\"per-emitted-frame identity and scale\",\"reference_shape\":\"identity=0, global_scale=1, other_scales=0; not fixed bone lengths\",\"joint_names\":";
 names(o,rest.names);o<<",\"parents\":";array(o,rest.parents);o<<",\"rest_local_translations\":";array(o,rest.local_translations);o<<",\"rest_local_rotations\":";array(o,rest.local_rotations);
 o<<"},\"smpl24\":{\"schema\":\"gemx.smpl24.v1\",\"joint_count\":24,\"root\":0,\"units\":\"metres\",\"handedness\":\"right\",\"quaternion_order\":\"wxyz\",\"space\":\"root-local, translation and anchor rotation removed\",\"anchor_basis\":\"Z-up, upstream SMPL base rotation removed\",\"reconstruction\":\"YtoZ*(soma[mapping]-soma[0])=rotate(anchor,smpl_local)\",\"quaternion_policy\":\"unit, canonical w>=0\",\"mapping\":";
 array(o,gemx::smpl_mapping);std::vector<std::string> sn(std::begin(gemx::smpl_names),std::end(gemx::smpl_names));o<<",\"joint_names\":";names(o,sn);
 o<<",\"parents\":[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21],\"reference_positions\":";array(o,smpl);o<<",\"reference_anchor\":";array(o,anchor);
 o<<",\"reference_shape\":\"same neutral identity and scale as soma77\",\"rest_local_translations\":";array(o,smpl_rest_local);
 o<<",\"rest_local_rotations\":";array(o,smpl_rest_rotations);
 o<<",\"rest_rotation_policy\":\"identity point-coordinate frames, not fitted SMPL joint rotations\",\"soma_y_to_z_matrix_row_major\":[1,0,0,0,0,-1,0,1,0]}";
 o<<",\"channel_spaces\":{\"positions\":\"soma77 gravity-aligned Y-up, root translation removed only\",\"local_rotations\":\"soma77 parent-local xyzw\",\"local_translations\":\"soma77 parent-local metres\",\"root_axis_angle\":\"SOMA gravity-aligned Y-up axis-angle radians\",\"root_translation\":\"zero, no continuous world trajectory\",\"camera_positions\":\"SOMA camera-oriented root-relative metres\",\"camera_translation\":\"add to camera_positions for perspective projection with f=max(width,height) and image-centre principal point\",\"keypoints\":\"source pixel x/y and 2D confidence, not 3D confidence\",\"box\":\"source pixel xyxy\",\"identity\":\"current emitted-frame SOMA identity coefficients\",\"scales\":\"current emitted-frame SOMA scale parameters\"}";
 o<<",\"channels\":{\"positions\":231,\"local_rotations\":308,\"local_translations\":231,\"smpl_joints\":72,\"smpl_anchor\":4,\"root_axis_angle\":3,\"root_translation\":3,\"camera_positions\":231,\"camera_translation\":3,\"keypoints\":231,\"box\":4,\"metrics\":9,\"identity\":45,\"scales\":69}}";
 return o.str();
}
}
gemx_status gemx_live_create(const char*g,const char*v,const char*y,const char*m,const char*b,uint32_t device,const char*expected,uint32_t threads,uint32_t window,uint32_t cadence,uint32_t policy,uint32_t precision,uint32_t temporal,int64_t gap,gemx_live**out,char*err,uint64_t cap){
 if(out)*out=nullptr;
 return gemx::boundary(err,cap,[&]{
  require(out&&g&&v&&y&&m&&b,"model paths, backend and output required");
  require(threads>=1&&threads<=8&&window>=2&&window<=120&&cadence>=1&&cadence<=30&&policy<=1&&precision<=1&&temporal==GEMX_LIVE_FRAME_INDEX&&gap>0,"invalid live configuration");
  if(precision==0&&std::string(b)=="Vulkan")for(const char*key:{"GGML_VK_DISABLE_F16","GGML_VK_DISABLE_COOPMAT","GGML_VK_DISABLE_COOPMAT2"}){const char*value=std::getenv(key);require(value&&std::string(value)=="1","strict Vulkan requires preconfigured GGML_VK_DISABLE_F16/COOPMAT/COOPMAT2=1");}
  auto p=std::make_unique<gemx_live>();p->window=window;p->cadence=cadence;p->selection=policy;p->precision=precision;p->threads=threads;p->gap=gap;
  gemx_session_config c{g,m,b,expected,device,threads,std::min(window,32u)};
  p->backend=b;p->module=m;p->device=device;p->gem_model=g;p->pose_model=v;p->detector_model=y;
  p->session=std::make_unique<gemx::session>(c,true);
  char message[512];gemx_vitpose*vp=nullptr;c.model_path=v;c.graph_cache_capacity=1;
  call(gemx_vitpose_create(&c,&vp,message,sizeof(message)),message);p->pose.reset(vp);
  gemx_yolox*yp=nullptr;c.model_path=y;call(gemx_yolox_create(&c,&yp,message,sizeof(message)),message);p->detector.reset(yp);
  p->definition=definition(*p);*out=p.release();
 });
}
void gemx_live_destroy(gemx_live*p){delete p;}
void gemx_live_result_destroy(gemx_live_result*r){delete r;}
gemx_status gemx_live_definition(gemx_live*p,char*out,uint64_t capacity,uint64_t*required,char*err,uint64_t cap){
 if(required)*required=0;
 return gemx::boundary(err,cap,[&]{require(p&&required&& (out||!capacity),"invalid definition buffer");std::lock_guard lock(p->mutex);*required=p->definition.size()+1;if(!out&&!capacity)return;require(capacity>=*required,"definition capacity too small");std::memcpy(out,p->definition.c_str(),*required);});
}
gemx_status gemx_live_reset(gemx_live*p,char*err,uint64_t cap){return gemx::boundary(err,cap,[&]{require(p,"pipeline required");std::lock_guard lock(p->mutex);p->state.reset();p->history.clear();});}
gemx_status gemx_live_result_info(const gemx_live_result*r,uint64_t*s,int64_t*t,uint64_t*e,uint64_t*track,uint32_t*status,uint32_t*flags,char*err,uint64_t cap){
 return gemx::boundary(err,cap,[&]{require(r&&s&&t&&e&&track&&status&&flags,"result and output pointers required");*s=r->sequence;*t=r->timestamp;*e=r->epoch;*track=r->track;*status=r->outcome;*flags=r->flags;});
}
gemx_status gemx_live_result_copy(const gemx_live_result*r,uint32_t channel,float*out,uint64_t capacity,uint64_t*required,char*err,uint64_t cap){
 if(required)*required=0;
 return gemx::boundary(err,cap,[&]{require(r&&channel<14&&required&&(out||!capacity),"invalid result buffer/channel");const auto&d=r->data[channel];*required=d.size();if(!out&&!capacity)return;require(capacity>=*required,"result capacity too small");std::copy(d.begin(),d.end(),out);});
}
gemx_status gemx_live_submit(gemx_live*p,const uint8_t*rgb,uint64_t capacity,uint32_t w,uint32_t h,uint64_t stride,uint64_t seq,int64_t time,const float*box,uint64_t box_count,uint64_t subject,gemx_live_result**out,char*err,uint64_t cap){
 if(out)*out=nullptr;
 return gemx::boundary(err,cap,[&]{
  require(p&&out&&rgb,"pipeline, RGB and result required");
  require(w>=8&&h>=8&&w<=32766&&h<=32766&&uint64_t(w)*h<=16000000,"invalid frame dimensions");
  require(stride>=uint64_t(w)*3&&capacity>=uint64_t(w)*3&&stride<=(capacity-uint64_t(w)*3)/(h-1),"invalid RGB capacity/stride");
  require((!box&&box_count==0)||(box&&box_count==4),"subject box must have four coordinates");
  if(box){for(size_t i=0;i<4;++i)require(std::isfinite(box[i]),"nonfinite box");require(box[0]>=0&&box[1]>=0&&box[2]<=w-1&&box[3]<=h-1&&box[2]>box[0]&&box[3]>box[1],"invalid subject box");}
  std::lock_guard lock(p->mutex);p->state.validate(seq,time);
  auto state=p->state;auto history=p->history;const auto dt=state.accepted?time-state.time:0;
  auto result=std::make_unique<gemx_live_result>();result->sequence=seq;result->timestamp=time;
  if(state.begin(seq,time,w,h,p->gap,box!=nullptr,subject)){history.clear();result->flags|=GEMX_LIVE_RESET;}
  result->data[GEMX_LIVE_METRICS].assign(9,0);auto&metrics=result->data[GEMX_LIVE_METRICS];metrics[5]=float(dt);metrics[8]=-1;
  auto mark=clock_type::now();auto stage=[&](size_t i){auto now=clock_type::now();metrics[i]=std::chrono::duration<float,std::milli>(now-mark).count();mark=now;};
  char message[512];gemx_rgb_frame frame{rgb,capacity,w,h,stride,{}};
  try{
   bool usable=true;
   if(box){if(!state.selected)++state.track;state.selected=true;state.confidence=-1;std::copy_n(box,4,state.box.begin());result->flags|=GEMX_LIVE_CALLER_BOX;}
   else if(!state.remaining){
    result->flags|=GEMX_LIVE_DETECTOR_RAN;
    std::vector<gemx_detection> ds(100);uint32_t count=0;
    call(gemx_yolox_detect(p->detector.get(),&frame,.5f,.45f,ds.data(),ds.size(),&count,message,sizeof(message)),message);ds.resize(count);state.detected=count;metrics[6]=float(count);
    // Preserve upstream largest-area selection before image clipping.
    if(p->selection==GEMX_LIVE_CONTINUITY)for(auto&d:ds)for(int k=0;k<4;++k)d.box[k]=std::clamp(d.box[k],0.f,float(k%2?h-1:w-1));
    int selected=state.choose(ds,p->selection);
    if(selected>=0){if(!state.selected&&p->selection==GEMX_LIVE_CONTINUITY)++state.track;state.selected=true;state.confidence=ds[selected].score;std::copy_n(ds[selected].box,4,state.box.begin());for(int k=0;k<4;++k)state.box[k]=std::clamp(state.box[k],0.f,float(k%2?h-1:w-1));state.remaining=p->cadence-1;}
    else if(p->selection==GEMX_LIVE_PARITY){state.box={0,0,float(w-1),float(h-1)};state.selected=false;state.confidence=-1;result->flags|=GEMX_LIVE_FULL_IMAGE;}
    else{
     if(state.selected||!history.empty()){state.reset();state.begin(seq,time,w,h,p->gap,false,0);result->flags|=GEMX_LIVE_RESET;}
     state.selected=false;state.remaining=0;history.clear();usable=false;result->outcome=selected==-2?GEMX_LIVE_AMBIGUOUS:GEMX_LIVE_LOST;
    }
   }else{--state.remaining;metrics[6]=float(state.detected);result->flags|=GEMX_LIVE_CROP_REUSED;}
   stage(0);
   if(usable)result->data[GEMX_LIVE_BOX].assign(state.box.begin(),state.box.end());
   metrics[8]=state.confidence;
   if(usable){
    observation current{};const auto&b=state.box;
    current.box={(b[0]+b[2])*.5f,(b[1]+b[3])*.5f,std::max(b[3]-b[1],(b[2]-b[0])/.75f)*1.2f};std::copy(current.box.begin(),current.box.end(),frame.box);
    call(gemx_vitpose_infer_rgb(p->pose.get(),&frame,1,current.keypoints.data(),231,message,sizeof(message)),message);stage(1);
    result->data[GEMX_LIVE_KEYPOINTS].assign(current.keypoints.begin(),current.keypoints.end());
    history.push_back(current);if(history.size()>p->window)history.pop_front();const uint32_t n=history.size();metrics[7]=float(n);
    if(n>=2){
     std::vector<float> kp(n*231),boxes(n*3),K(n*9),angular(n*6);
     for(uint32_t i=0;i<n;++i){std::copy(history[i].keypoints.begin(),history[i].keypoints.end(),kp.begin()+i*231);std::copy(history[i].box.begin(),history[i].box.end(),boxes.begin()+i*3);
      const float f=float(std::max(w,h));const std::array<float,9> k{f,0,w*.5f,0,f,h*.5f,0,0,1};std::copy(k.begin(),k.end(),K.begin()+i*9);angular[i*6]=angular[i*6+4]=1;}
     gemx_sequence_view input{n,kp.data(),boxes.data(),K.data(),nullptr,angular.data()};
     std::vector<float> body(n*228),identity(n*45),scales(n*69),oc(n*3),tc(n*3),ow(n*3),tw(n*3),pred(n*585),camera(n*3);
     gemx_motion_view motion{n,body.data(),identity.data(),scales.data(),oc.data(),tc.data(),ow.data(),tw.data()};
     p->session->infer(input,pred.data(),camera.data());p->session->decode(input,pred.data(),camera.data(),motion);stage(2);
     const auto last=n-1;std::array<float,3> origin{};
     gemx_motion_view newest{1,body.data()+last*228,identity.data()+last*45,scales.data()+last*69,oc.data()+last*3,tc.data()+last*3,ow.data()+last*3,origin.data()};
     auto sk=p->session->skeleton(newest);stage(3);
     auto&d=result->data;d[0]=std::move(sk.positions);d[1]=std::move(sk.local_rotations);d[2]=std::move(sk.local_translations);
     d[5].assign(newest.global_orient_world,newest.global_orient_world+3);d[6].assign(3,0);
     d[12].assign(newest.identity_coeffs,newest.identity_coeffs+45);d[13].assign(newest.scale_params,newest.scale_params+69);
     newest.global_orient_world=newest.global_orient_camera;auto cs=p->session->skeleton(newest);d[7]=std::move(cs.positions);d[8].assign(newest.translation_camera,newest.translation_camera+3);stage(4);
     d[3].resize(72);d[4].resize(4);call(gemx_soma_to_smpl(d[0].data(),231,d[5].data(),3,d[3].data(),72,d[4].data(),4,message,sizeof(message)),message);
     result->outcome=GEMX_LIVE_POSE;
    }
   }
  }catch(const std::invalid_argument&e){p->state.reset();p->history.clear();throw std::runtime_error(e.what());}
  catch(...){p->state.reset();p->history.clear();throw;}
  result->epoch=state.epoch;result->track=state.track;p->state=state;p->history=std::move(history);*out=result.release();
 });
}

gemx_status gemx_live_topology(gemx_live*p,int32_t*out,uint64_t count,char*err,uint64_t cap){return gemx::boundary(err,cap,[&]{require(p&&out&&count==77,"topology needs 77 parents");std::lock_guard lock(p->mutex);std::copy(p->parents.begin(),p->parents.end(),out);});}
