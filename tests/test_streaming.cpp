#include "../src/live_state.hpp"
#include <fstream>
#include <iostream>
#include <limits>
using gemx::require;
void rejects(auto f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}require(caught,"expected rejection");}
int main(int argc,char**argv){try{
 require(argc==2,"fixture argument required");
 gemx::live_state a,b;
 require(a.begin(8,9007199254740993LL,640,480,100000,false,0),"initial reset");
 a.validate(10,9007199254740994LL); // exact int64, above JavaScript safe integer
 rejects([&]{a.validate(8,9007199254740994LL);});
 rejects([&]{a.validate(9,9007199254740993LL);});
 rejects([&]{a.validate(9,-1);});
 require(a.seq==8&&a.time==9007199254740993LL&&a.epoch==1,"rejection mutated state");
 require(!a.begin(10,9007199254740994LL,640,480,100000,false,0),"sequence gaps allowed");
 require(a.begin(11,9007199254840995LL,640,480,100000,false,0)&&a.epoch==2,"time gap reset");
 require(a.begin(12,9007199254840996LL,641,480,100000,false,0)&&a.epoch==3,"resolution reset");
 require(a.begin(13,9007199254840997LL,641,480,100000,true,5),"selection mode reset");
 require(!a.begin(14,9007199254840998LL,641,480,100000,true,5),"same subject reset");
 require(a.begin(15,9007199254840999LL,641,480,100000,true,6),"subject reset");
 a.reset();a.validate(0,0);require(a.begin(0,0,640,480,100000,false,0),"explicit restart");
 require(!b.accepted&&b.epoch==1,"instance isolation");
 gemx_detection left{},right{},overlap{};
 std::copy_n(std::array<float,4>{0,0,100,100}.data(),4,left.box);
 std::copy_n(std::array<float,4>{200,0,350,150}.data(),4,right.box);
 std::copy_n(std::array<float,4>{10,5,105,105}.data(),4,overlap.box);
 require(a.choose({},1)==-1,"loss");require(a.choose({left,right},1)==-2,"initial ambiguity");
 require(a.choose({left,right},0)==1,"largest parity");
 require(a.choose({left},1)==0,"unique acquisition");a.selected=true;a.box={0,0,100,100};
 require(a.choose({right,overlap},1)==1,"association");
 require(a.choose({left,overlap},1)==-2,"ambiguous association");
 require(a.choose({right},1)==-1,"must not switch subjects");a.reset();
 require(a.choose({right},1)==0,"reacquisition after reset");
 std::ifstream in(argv[1]);int n=0;in>>n;require(n>=12&&n<=1000,"fixture cases");
 float maxp=0,maxq=0;char error[256];
 for(int k=0;k<n;++k){
  std::array<float,231> joints{};std::array<float,3> root{};std::array<float,72> expected{},actual{};std::array<float,4> eq{},q{};
  for(auto&x:joints)in>>x;for(auto&x:root)in>>x;for(auto&x:expected)in>>x;for(auto&x:eq)in>>x;require(bool(in),"truncated fixture");
  require(gemx_soma_to_smpl(joints.data(),231,root.data(),3,actual.data(),72,q.data(),4,error,sizeof(error))==GEMX_OK,error);
  float dot=0;for(int i=0;i<4;++i)dot+=q[i]*eq[i];
  for(int i=0;i<4;++i)maxq=std::max(maxq,std::abs(q[i]-(dot<0?-eq[i]:eq[i])));
  for(int i=0;i<72;++i)maxp=std::max(maxp,std::abs(actual[i]-expected[i]));
  require(maxp<1e-5f&&maxq<1e-6f,"upstream conversion mismatch");
  auto saved=actual;joints[6]=std::numeric_limits<float>::quiet_NaN();
  require(gemx_soma_to_smpl(joints.data(),231,root.data(),3,actual.data(),72,q.data(),4,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&actual==saved,"NaN/transactionality");
  require(gemx_soma_to_smpl(nullptr,231,root.data(),3,actual.data(),72,q.data(),4,error,sizeof(error))==GEMX_INVALID_ARGUMENT,"null input");
  require(gemx_soma_to_smpl(joints.data(),230,root.data(),3,actual.data(),71,q.data(),4,error,sizeof(error))==GEMX_INVALID_ARGUMENT,"short capacity");
 }
 uint64_t needed=9;require(gemx_live_result_copy(nullptr,0,nullptr,0,&needed,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&needed==0,"invalid result");
 gemx_live*p=reinterpret_cast<gemx_live*>(1);
 require(gemx_live_create("a","b","c","d","CPU",0,nullptr,9,30,1,0,0,0,100,&p,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&!p,"invalid config");
 require(gemx_live_create("a","b","c","d","CPU",0,nullptr,8,30,1,0,0,1,100,&p,error,sizeof(error))==GEMX_INVALID_ARGUMENT&&!p,"unsupported temporal policy");
 gemx_live_destroy(nullptr);gemx_live_result_destroy(nullptr);
 std::cout<<n<<" upstream fixtures: max_position="<<maxp<<" max_quaternion="<<maxq<<"; lifecycle state passed\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
