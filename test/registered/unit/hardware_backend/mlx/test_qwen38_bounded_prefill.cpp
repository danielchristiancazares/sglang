#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "mlx/mlx.h"
#include "qwen38_engine.h"
namespace mx = mlx::core;
namespace native = sglang::mlx_qwen38;
mx::array Values(const mx::Shape& shape, int seed, float scale) {
 std::size_t count=1;for(int dimension:shape) count*=dimension;
 std::vector<float> values(count);
 std::uint32_t state=seed;
 for(auto& value:values){state=state*1664525U+1013904223U;value=float(int((state>>16)%2001)-1000)*scale;}
 return mx::astype(mx::array(values.data(),shape,mx::float32),native::activation_dtype());
}
mx::array Oracle(const mx::array& q, const mx::array& k, const mx::array& v, int prefix) {
 const int count=q.shape()[2], length=prefix+count;
 const auto qf=mx::astype(q,mx::float32);
 const auto kf=mx::astype(mx::slice(k,{0,0,0,0},{1,4,length,256}),mx::float32);
 const auto vf=mx::astype(mx::slice(v,{0,0,0,0},{1,4,length,256}),mx::float32);
 const auto grouped=mx::reshape(qf,{1,4,6*count,256});
 const auto scores=mx::matmul(grouped,mx::swapaxes(kf,-1,-2))*0.0625f;
 std::vector<int> limits(6*count);
 for(int row=0;row<6*count;++row) limits[row]=prefix+row%count;
 const auto mask=mx::less_equal(mx::reshape(mx::arange(length,mx::int32),{1,1,1,length}),
    mx::array(limits.data(),{1,1,6*count,1},mx::int32));
 const auto probabilities=mx::softmax(mx::where(mask,scores,mx::array(-std::numeric_limits<float>::infinity())),-1,true);
 return mx::reshape(mx::matmul(probabilities,vf),q.shape());
}
void Check(int count,int prefix,int capacity,bool quantized,float sharpness=1) {
 const int length=count+prefix;
 auto q=Values({1,24,count,256},11,0.001f*sharpness);
 auto k=Values({1,4,capacity,256},29,0.001f);
 auto v=Values({1,4,capacity,256},43,0.001f);
 std::vector<mx::array> qk,qv;
 if(quantized) {
  qk=mx::quantize(k,64,8,"affine");qv=mx::quantize(v,64,8,"affine");
  k=mx::dequantize(qk[0],qk[1],qk[2],64,8,"affine",std::nullopt,native::activation_dtype());
  v=mx::dequantize(qv[0],qv[1],qv[2],64,8,"affine",std::nullopt,native::activation_dtype());
 }
 const auto expected=Oracle(q,k,v,prefix);mx::eval(expected);
 if(length<capacity) {
  if(quantized) {
   const auto poison=mx::full({1,4,capacity-length,4},std::numeric_limits<float>::quiet_NaN(),native::activation_dtype());
   for(int field:{1,2}) {qk[field]=mx::slice_update(qk[field],poison,{0,0,length,0},{1,4,capacity,4});qv[field]=mx::slice_update(qv[field],poison,{0,0,length,0},{1,4,capacity,4});}
  } else {
   const auto poison=mx::full({1,4,capacity-length,256},std::numeric_limits<float>::quiet_NaN(),native::activation_dtype());
   k=mx::slice_update(k,poison,{0,0,length,0},{1,4,capacity,256});v=mx::slice_update(v,poison,{0,0,length,0},{1,4,capacity,256});
  }
 }
 const auto actual=quantized ? native::fixed_q8_attention(q,qk[0],qk[1],qk[2],qv[0],qv[1],qv[2],prefix,length)
    : native::fixed_prefill_attention(q,k,v,prefix,length);
 const auto difference=mx::abs(mx::astype(actual,mx::float32)-expected);
 const auto maximum=mx::max(difference), magnitude=mx::max(mx::abs(expected)), finite=mx::all(mx::isfinite(actual));
 mx::eval(actual,maximum,magnitude,finite);
 const float tolerance=native::activation_dtype()==mx::float16 ? 0.003f : 0.025f;
 std::cout<<"queries="<<count<<" prefix="<<prefix<<" capacity="<<capacity<<" q8="<<quantized<<" sharpness="<<sharpness
   <<" max_abs="<<maximum.item<float>()<<" magnitude="<<magnitude.item<float>()<<" finite="<<finite.item<bool>()
   <<" peak_bytes="<<mx::get_peak_memory()<<std::endl;
 if(actual.shape()!=q.shape()||!finite.item<bool>()||maximum.item<float>()>tolerance*(1+magnitude.item<float>()))
  throw std::runtime_error("bounded prefill differs from independent causal FP32 oracle");
}
int main(int argc,char** argv) {
 try {
  setenv("SGLANG_MLX_NATIVE_PREFILL_SDPA","1",1);
  if(argc==2&&std::string(argv[1])=="capacity") {
   const auto q=Values({1,24,1024,256},11,0.0001f),k=Values({1,4,131072,256},29,0.001f),v=Values({1,4,131072,256},43,0.001f);
   auto qk=mx::quantize(k,64,8,"affine"),qv=mx::quantize(v,64,8,"affine");
   const auto output=native::fixed_q8_attention(q,qk[0],qk[1],qk[2],qv[0],qv[1],qv[2],130048,131072);
   const auto finite=mx::all(mx::isfinite(output));mx::eval(output,finite);mx::synchronize();
   std::cout<<"queries=1024 active=131072 finite="<<finite.item<bool>()<<" peak_bytes="<<mx::get_peak_memory()<<std::endl;
   return finite.item<bool>()?0:1;
  }
  for(bool q8:{false,true}) {
   for(int count:{16,17,64,129,256}) Check(count,67,512,q8,8);
   Check(1024,0,1024,q8);
   Check(17,131040,131072,q8,8);
   Check(256,16121,16384,q8,2);
   Check(65,65501,131072,q8,1);
  }
  std::cout<<"bounded_prefill_independent_causal_oracle=pass\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
