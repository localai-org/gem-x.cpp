// Optional resident API integration test; requires the three local live GGUFs.
#include "gemx_stream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace {
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
char error[512];void check(gemx_status s){require(s==GEMX_OK,error);}
using pipeline=std::unique_ptr<gemx_live,decltype(&gemx_live_destroy)>;
using result=std::unique_ptr<gemx_live_result,decltype(&gemx_live_result_destroy)>;
struct image {uint32_t w=0,h=0,stride=0;std::vector<uint8_t> rgb;std::array<float,4> box{};};
image load(const char*path){
 std::ifstream f(path,std::ios::binary);char magic[8];image im;
 f.read(magic,8);require(f&&std::string(magic,8)=="S3DIMG01","bad frame magic");
 f.read(reinterpret_cast<char*>(&im.w),4);f.read(reinterpret_cast<char*>(&im.h),4);f.read(reinterpret_cast<char*>(&im.stride),4);f.read(reinterpret_cast<char*>(im.box.data()),16);f.ignore(16);
 require(f&&im.w>=8&&im.h>=8&&im.w<=32766&&im.h<=32766&&uint64_t(im.w)*im.h<=16000000&&im.stride==im.w*3,"bad frame dimensions");
 im.rgb.resize(uint64_t(im.stride)*im.h);f.read(reinterpret_cast<char*>(im.rgb.data()),im.rgb.size());require(f&&f.peek()==EOF,"bad frame payload");return im;
}
std::vector<float> channel(const result&r,uint32_t c){uint64_t n;check(gemx_live_result_copy(r.get(),c,nullptr,0,&n,error,sizeof(error)));std::vector<float> v(n);check(gemx_live_result_copy(r.get(),c,v.data(),v.size(),&n,error,sizeof(error)));return v;}
struct info {uint64_t seq,epoch,track;int64_t time;uint32_t status,flags;};
info details(const result&r){info v;check(gemx_live_result_info(r.get(),&v.seq,&v.time,&v.epoch,&v.track,&v.status,&v.flags,error,sizeof(error)));return v;}
result submit(pipeline&p,const image&im,uint64_t seq,int64_t t,uint64_t subject=7){
 auto box=im.box;if(box[2]<=box[0])box={0,0,float(im.w-1),float(im.h-1)};gemx_live_result*r=nullptr;
 check(gemx_live_submit(p.get(),im.rgb.data(),im.rgb.size(),im.w,im.h,im.stride,seq,t,box.data(),4,subject,&r,error,sizeof(error)));return {r,gemx_live_result_destroy};
}
using vec=std::array<double,3>;using quat=std::array<double,4>; // WXYZ for independent test FK
quat mul(quat a,quat b){return {a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};}
vec rotate(quat q,vec v){auto a=mul(mul(q,{0,v[0],v[1],v[2]}),{q[0],-q[1],-q[2],-q[3]});return {a[1],a[2],a[3]};}
void fk(const result&r,const std::array<int32_t,77>&parents){
 auto p=channel(r,0),q=channel(r,1),t=channel(r,2);require(p.size()==231&&q.size()==308&&t.size()==231,"channel lengths");
 std::array<vec,77> positions{};std::array<quat,77> rotations{};
 for(size_t i=0;i<77;++i){int parent=parents[i];require(parent>=-1&&parent<int(i),"topology order");quat local{q[i*4+3],q[i*4],q[i*4+1],q[i*4+2]};vec trans{t[i*3],t[i*3+1],t[i*3+2]};
  if(parent<0){rotations[i]=local;positions[i]=trans;}else{rotations[i]=mul(rotations[parent],local);auto offset=rotate(rotations[parent],trans);for(int k=0;k<3;++k)positions[i][k]=positions[parent][k]+offset[k];}
  for(int k=0;k<3;++k)require(std::abs(positions[i][k]-p[i*3+k])<1e-5,"local transforms do not reconstruct emitted pose");
 }
}
}
int main(int argc,char**argv){try{
 require(argc==9,"usage: test GEM VITPOSE YOLOX MODULE BACKEND DEVICE FRAME1.input FRAME2.input");
 auto create=[&](uint32_t window=2){gemx_live*p=nullptr;check(gemx_live_create(argv[1],argv[2],argv[3],argv[4],argv[5],std::stoul(argv[6]),nullptr,8,window,5,1,0,0,100000,&p,error,sizeof(error)));return pipeline(p,gemx_live_destroy);};
 auto a=create(),b=create();auto first=load(argv[7]),second=load(argv[8]);
 uint64_t bytes;check(gemx_live_definition(a.get(),nullptr,0,&bytes,error,sizeof(error)));require(bytes>1000&&bytes<100000,"definition size");
 std::vector<char> definition(bytes);check(gemx_live_definition(a.get(),definition.data(),bytes,&bytes,error,sizeof(error)));require(std::string(definition.data()).find("gemx.smpl24.v1")!=std::string::npos,"missing SMPL definition");
 require(gemx_live_definition(a.get(),definition.data(),1,&bytes,error,sizeof(error))==GEMX_INVALID_ARGUMENT,"definition capacity");
 std::array<int32_t,77> parents;check(gemx_live_topology(a.get(),parents.data(),77,error,sizeof(error)));
 constexpr int64_t origin=9007199254740993LL;
 auto aw=submit(a,first,10,origin);require(details(aw).status==0&&details(aw).time==origin&&(details(aw).flags&1),"warmup timing/reset");
 auto ap=submit(a,second,20,origin+20000);require(details(ap).status==1&&details(ap).epoch==1,"second pose");fk(ap,parents);
 auto saved=channel(ap,0),shape=channel(ap,12),scales=channel(ap,13);
 // All rejected inputs must leave the next accepted frame's history untouched.
 gemx_live_result*bad=nullptr;float box[]={0,0,float(first.w-1),float(first.h-1)};
 require(gemx_live_submit(a.get(),first.rgb.data(),first.rgb.size(),first.w,first.h,first.stride,20,origin+30000,box,4,7,&bad,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&!bad,"duplicate sequence");
 require(gemx_live_submit(a.get(),first.rgb.data(),first.rgb.size(),first.w,first.h,first.stride,21,origin+20000,box,4,7,&bad,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&!bad,"duplicate time");
 require(gemx_live_submit(a.get(),first.rgb.data(),1,first.w,first.h,first.stride,30,origin+40000,box,4,7,&bad,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&!bad,"RGB capacity");
 require(gemx_live_submit(a.get(),first.rgb.data(),first.rgb.size(),first.w,first.h,UINT64_MAX,30,origin+40000,box,4,7,&bad,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&!bad,"stride overflow");
 float short_buffer[3]={123,123,123};uint64_t needed;
 require(gemx_live_result_copy(ap.get(),0,short_buffer,3,&needed,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&needed==231&&short_buffer[0]==123,"output capacity");
 auto bw=submit(b,first,10,origin);require(details(bw).status==0&&details(bw).epoch==1,"instance warmup isolation");
 auto bp=submit(b,second,20,origin+20000);require(channel(bp,0)==saved,"independent inference mismatch");
 auto ax=submit(a,second,30,origin+40000),bx=submit(b,second,30,origin+40000);require(channel(ax,0)==channel(bx,0),"rejected inputs changed history");fk(ax,parents);
 require(channel(ax,12)!=shape||channel(ax,13)!=scales,"fixture did not exercise varying identity/scale");
 auto gap=submit(a,first,31,origin+200001);require(details(gap).status==0&&details(gap).epoch==2&&(details(gap).flags&1),"gap reset");
 auto subject=submit(a,first,32,origin+220001,8);require(details(subject).status==0&&details(subject).epoch==3,"subject reset");
 image smaller{8,8,24,std::vector<uint8_t>(192)};auto size=submit(a,smaller,33,origin+240001,8);require(details(size).status==0&&details(size).epoch==4,"resolution reset");
 check(gemx_live_reset(a.get(),error,sizeof(error)));auto restarted=submit(a,first,0,0);require(details(restarted).epoch==5&&details(restarted).status==0,"explicit reset");
 auto unaffected=submit(b,second,40,origin+60000);require(details(unaffected).epoch==1&&details(unaffected).status==1,"reset leaked to other instance");
 a.reset();b.reset();require(channel(ap,0)==saved,"result lifetime after reset/destroy");
 auto c=create(120);require(details(submit(c,first,0,0)).epoch==1,"repeated creation state");
 auto automatic=[&](const image&im,uint64_t seq){gemx_live_result*r=nullptr;check(gemx_live_submit(c.get(),im.rgb.data(),im.rgb.size(),im.w,im.h,im.stride,seq,int64_t(seq)*20000,nullptr,0,0,&r,error,sizeof(error)));return result(r,gemx_live_result_destroy);};
 auto lost=automatic(smaller,1);require(details(lost).status==GEMX_LIVE_LOST&&(details(lost).flags&GEMX_LIVE_DETECTOR_RAN),"automatic loss");
 auto still_lost=automatic(smaller,2);require(details(still_lost).status==GEMX_LIVE_LOST&&(details(still_lost).flags&GEMX_LIVE_DETECTOR_RAN),"loss must force detection");
 auto acquired=automatic(first,3);require(details(acquired).status==GEMX_LIVE_WARMUP&&details(acquired).track>details(lost).track,"reacquisition must warm up");
 auto reused=automatic(first,4);require(details(reused).status==GEMX_LIVE_POSE&&(details(reused).flags&GEMX_LIVE_CROP_REUSED)&&!(details(reused).flags&GEMX_LIVE_DETECTOR_RAN),"crop cadence/freshness");
 std::cout<<"two resident instances, ownership, timing, invalid buffers, resets, loss/reacquisition, cadence and frame-specific FK passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
