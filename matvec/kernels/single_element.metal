#include <metal_stdlib>
using namespace metal;
kernel void E(device float* data0_1, device float* data1_1, device float* data2_1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  float val0 = (*(data1_1+0));
  float val1 = (*(data2_1+0));
  *(data0_1+0) = (val0*val1);
}