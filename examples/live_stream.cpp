// A direct native client: no worker process, HTTP, Python or Body model.
#include "gemx_stream.h"
#include <fstream>
#include <charconv>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
int main(int argc,char**argv){try{
 if(argc!=8)throw std::runtime_error("usage: gemx-live-stream GEM.gguf VITPOSE.gguf YOLOX.gguf MODULE CPU|Vulkan DEVICE manifest.txt");
 uint32_t device=0;const char*end=argv[6]+std::strlen(argv[6]);auto parsed=std::from_chars(argv[6],end,device);
 if(parsed.ec!=std::errc{}||parsed.ptr!=end)throw std::runtime_error("invalid device index");
 char error[512]{};auto check=[&](gemx_status s){if(s!=GEMX_OK)throw std::runtime_error(error);};
 gemx_live*raw=nullptr;
 check(gemx_live_create(argv[1],argv[2],argv[3],argv[4],argv[5],device,nullptr,
  8,30,1,GEMX_LIVE_CONTINUITY,GEMX_LIVE_STRICT_F32,GEMX_LIVE_FRAME_INDEX,2000000,&raw,error,sizeof(error)));
 std::unique_ptr<gemx_live,decltype(&gemx_live_destroy)> pipeline(raw,gemx_live_destroy);
 uint64_t bytes=0;check(gemx_live_definition(raw,nullptr,0,&bytes,error,sizeof(error)));
 std::vector<char> definition(bytes);check(gemx_live_definition(raw,definition.data(),bytes,&bytes,error,sizeof(error)));
 std::cout<<definition.data()<<'\n';
 // Manifest: width height on first line, then sequence source_us raw_rgb_path.
 // RGB files are tightly packed; paths are relative to the working directory.
 std::ifstream input(argv[7]);uint32_t w=0,h=0;input>>w>>h;
 if(!input||w<8||h<8||w>32766||h>32766||uint64_t(w)*h>16000000)throw std::runtime_error("invalid manifest dimensions");
 uint64_t sequence;int64_t time;std::string path,sequence_text;
 while(input>>sequence_text){
  auto parsed_sequence=std::from_chars(sequence_text.data(),sequence_text.data()+sequence_text.size(),sequence);
  if(parsed_sequence.ec!=std::errc{}||parsed_sequence.ptr!=sequence_text.data()+sequence_text.size())throw std::runtime_error("invalid manifest sequence");
  if(!(input>>time>>path))throw std::runtime_error("invalid manifest row");
  std::vector<uint8_t> rgb(uint64_t(w)*h*3);std::ifstream frame(path,std::ios::binary);
  if(!frame.read(reinterpret_cast<char*>(rgb.data()),rgb.size())||frame.peek()!=EOF)throw std::runtime_error("RGB file size mismatch");
  gemx_live_result*r=nullptr;check(gemx_live_submit(raw,rgb.data(),rgb.size(),w,h,uint64_t(w)*3,sequence,time,nullptr,0,0,&r,error,sizeof(error)));
  std::unique_ptr<gemx_live_result,decltype(&gemx_live_result_destroy)> result(r,gemx_live_result_destroy);
  uint64_t seq,epoch,track;int64_t timestamp;uint32_t status,flags;
  check(gemx_live_result_info(r,&seq,&timestamp,&epoch,&track,&status,&flags,error,sizeof(error)));
  std::cout<<"frame "<<seq<<" source_us "<<timestamp<<" epoch "<<epoch<<" track "<<track<<" status "<<status<<" flags "<<flags;
  if(status==GEMX_LIVE_POSE){
   float soma[231],root[3],smpl[72],anchor[4];uint64_t count;
   check(gemx_live_result_copy(r,GEMX_LIVE_POSITIONS,soma,231,&count,error,sizeof(error)));
   check(gemx_live_result_copy(r,GEMX_LIVE_ROOT_AXIS_ANGLE,root,3,&count,error,sizeof(error)));
   // Same standalone adapter works on a previously recorded decoded SOMA pose.
   check(gemx_soma_to_smpl(soma,231,root,3,smpl,72,anchor,4,error,sizeof(error)));
   std::cout<<" anchor_wxyz";for(float x:anchor)std::cout<<' '<<x;
   std::cout<<" smpl24";for(float x:smpl)std::cout<<' '<<x;
  }
  std::cout<<'\n';
 }
 if(!input.eof())throw std::runtime_error("invalid manifest sequence");
 check(gemx_live_reset(raw,error,sizeof(error))); // ready for another stream origin
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
