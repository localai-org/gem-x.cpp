#pragma once
#include "gemx_stream.h"
#include "internal.hpp"
#include <array>
#include <algorithm>
#include <vector>
namespace gemx {
using box4=std::array<float,4>;
inline float iou(const box4&a,const box4&b){
 float x=std::max(0.f,std::min(a[2],b[2])-std::max(a[0],b[0]));
 float y=std::max(0.f,std::min(a[3],b[3])-std::max(a[1],b[1]));
 float intersection=x*y,area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection;
 return area>0?intersection/area:0;
}
struct live_state {
 uint64_t epoch=1,track=0,seq=0,subject=0;
 int64_t time=0;uint32_t width=0,height=0,remaining=0,detected=0;
 bool accepted=false,selected=false,caller=false,pending_reset=true;
 box4 box{};float confidence=-1;
 void reset(){++epoch;accepted=false;selected=false;remaining=0;pending_reset=true;width=height=0;confidence=-1;box={};detected=0;}
 void validate(uint64_t s,int64_t t)const{require(t>=0,"negative source time");require(!accepted || (s>seq && t>time),"sequence and source time must increase");}
 bool begin(uint64_t s,int64_t t,uint32_t w,uint32_t h,int64_t gap,bool supplied,uint64_t id){
  bool reset_now=accepted && (t-time>gap || w!=width || h!=height || caller!=supplied || (supplied && subject!=id));
  if(reset_now)reset();
  bool clear=pending_reset;pending_reset=false;
  accepted=true;seq=s;time=t;width=w;height=h;caller=supplied;subject=id;
  return clear;
 }
 // -1 lost, -2 ambiguous; continuity only follows unique IoU>=.3 matches.
 int choose(const std::vector<gemx_detection>& ds,uint32_t policy)const{
  if(ds.empty())return -1;
  if(policy==GEMX_LIVE_PARITY){int best=0;float area=-1;for(size_t i=0;i<ds.size();++i){const auto*b=ds[i].box;float a=std::max(0.f,b[2]-b[0])*std::max(0.f,b[3]-b[1]);if(a>area){area=a;best=int(i);}}return best;}
  int match=-1,count=0;
  for(size_t i=0;i<ds.size();++i){box4 b;std::copy_n(ds[i].box,4,b.begin());if(!selected || iou(box,b)>=.3f){match=int(i);++count;}}
  return count==1?match:count==0?-1:-2;
 }
};
inline constexpr std::array<uint32_t,24> smpl_mapping{0,67,72,1,68,73,2,69,74,3,70,75,4,11,39,6,12,40,13,41,14,42,24,52};
inline constexpr const char* smpl_names[]={"pelvis","left_hip","right_hip","spine1","left_knee","right_knee","spine2","left_ankle","right_ankle","spine3","left_foot","right_foot","neck","left_collar","right_collar","head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hand","right_hand"};
inline constexpr const char* soma_mapping_names[]={"Hips","LeftLeg","RightLeg","Spine1","LeftShin","RightShin","Spine2","LeftFoot","RightFoot","Chest","LeftToeBase","RightToeBase","Neck1","LeftShoulder","RightShoulder","Head","LeftArm","RightArm","LeftForeArm","RightForeArm","LeftHand","RightHand","LeftHandMiddle1","RightHandMiddle1"};
}
