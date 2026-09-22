// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
// SOMA mapping and basis convention follow NVIDIA's Apache-2.0 live bridge.
// Pinned source and independent fixtures: reference/streaming-sources.json.
#include "gemx_stream.h"
#include "live_state.hpp"
#include <array>
#include <cmath>
namespace {
using quat=std::array<float,4>;
quat multiply(const quat&a,const quat&b){return {
 a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
 a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
 a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
 a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]};}
}
gemx_status gemx_soma_to_smpl(const float*j,uint64_t n,const float*a,uint64_t an,float*out,uint64_t on,float*qout,uint64_t qn,char*err,uint64_t cap){
 return gemx::boundary(err,cap,[&]{
  gemx::require(j&&a&&out&&qout&&n==231&&an==3&&on==72&&qn==4,"SMPL conversion counts must be 231/3/72/4");
  for(size_t i=0;i<231;++i)gemx::require(std::isfinite(j[i])&&std::abs(j[i])<=1e6f,"invalid joint coordinate");
  for(int i=0;i<3;++i)gemx::require(std::isfinite(a[i])&&std::abs(a[i])<=1e6f,"invalid root rotation");
  const float length=std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
  const float scale=length<1e-6f?.5f-length*length/48.f:std::sin(length*.5f)/length;
  const float half=std::sqrt(.5f);
  quat q=multiply(multiply({half,half,0,0},{std::cos(length*.5f),a[0]*scale,a[1]*scale,a[2]*scale}),{.5f,-.5f,-.5f,-.5f});
  float norm=0;for(float x:q)norm+=x*x;norm=std::sqrt(norm);for(float&x:q)x/=norm;
  if(q[0]<0)for(float&x:q)x=-x;
  std::array<float,72> local{};
  for(size_t k=0;k<24;++k){
   auto i=gemx::smpl_mapping[k]*3;
   const float x=j[i]-j[0],y=-(j[i+2]-j[2]),z=j[i+1]-j[1];
   // Rotate by conjugate(q), using two cross products.
   const float ux=-q[1],uy=-q[2],uz=-q[3];
   const float tx=2*(uy*z-uz*y),ty=2*(uz*x-ux*z),tz=2*(ux*y-uy*x);
   local[k*3]=x+q[0]*tx+uy*tz-uz*ty;
   local[k*3+1]=y+q[0]*ty+uz*tx-ux*tz;
   local[k*3+2]=z+q[0]*tz+ux*ty-uy*tx;
  }
  std::copy(local.begin(),local.end(),out);std::copy(q.begin(),q.end(),qout);
 });
}
