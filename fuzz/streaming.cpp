#include "gemx_stream.h"
#include "../src/live_state.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t*data,size_t size){
 if(size<16)return 0;
 std::array<float,231> joints{};std::array<float,3> root{};std::array<float,72> local{};std::array<float,4> q{};
 size_t copied=std::min(size-16,sizeof(joints));std::memcpy(joints.data(),data+16,copied);
 std::memcpy(root.data(),data,12);char error[256];
 gemx_soma_to_smpl(joints.data(),data[12]&1?231:data[12],root.data(),3,local.data(),data[13]&1?72:data[13]%73,q.data(),4,error,sizeof(error));
 gemx::live_state s;
 for(size_t i=0;i+15<size;i+=16){
  uint64_t seq;int64_t t;std::memcpy(&seq,data+i,8);std::memcpy(&t,data+i+8,8);
  try{s.validate(seq,t);s.begin(seq,t,8+data[i],8+data[i+1],100000,data[i+2]&1,data[i+3]);}catch(const std::invalid_argument&){}
  if(data[i+4]&1)s.reset();
 }
 return 0;
}
