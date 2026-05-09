#include <metal_stdlib>
using namespace metal;
kernel void r_8016_8_4_4_512(device float* data0_128256, device float* data1_4096, device float* data2_525336576, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  threadgroup __attribute__((aligned(16))) float temp0[128];
  float acc0[4];
  float acc1[4];
  int gidx0 = gid.x; /* 8016 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 4 */
  int alu0 = (lidx1+(gidx0<<4));
  *(acc0+0) = 0.0f;
  *(acc0+1) = 0.0f;
  *(acc0+2) = 0.0f;
  *(acc0+3) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 512; Ridx0++) {
    float val0 = (*(data1_4096+(lidx0+(Ridx0<<3))));
    int alu5 = (alu0+(lidx0*128256)+(Ridx0*1026048));
    float val1 = (*(data2_525336576+alu5));
    float val2 = (*(data2_525336576+(alu5+4)));
    float val3 = (*(data2_525336576+(alu5+8)));
    float val4 = (*(data2_525336576+(alu5+12)));
    *(acc0+0) = ((*(acc0+0))+(val0*val1));
    *(acc0+1) = ((*(acc0+1))+(val0*val2));
    *(acc0+2) = ((*(acc0+2))+(val0*val3));
    *(acc0+3) = ((*(acc0+3))+(val0*val4));
  }
  int alu11 = (lidx1<<5);
  *((threadgroup __attribute__((aligned(16))) float4*)((temp0+((lidx0<<2)+alu11)))) = float4((*(acc0+0)),(*(acc0+1)),(*(acc0+2)),(*(acc0+3)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  *(acc1+0) = 0.0f;
  *(acc1+1) = 0.0f;
  *(acc1+2) = 0.0f;
  *(acc1+3) = 0.0f;
  for (int Ridx102 = 0; Ridx102 < 8; Ridx102++) {
    float4 val5 = (*((threadgroup __attribute__((aligned(16))) float4*)((temp0+(alu11+(Ridx102<<2))))));
    *(acc1+0) = ((*(acc1+0))+val5.x);
    *(acc1+1) = ((*(acc1+1))+val5.y);
    *(acc1+2) = ((*(acc1+2))+val5.z);
    *(acc1+3) = ((*(acc1+3))+val5.w);
  }
  bool alu23 = (lidx0==0);
  if (alu23) {
    *(data0_128256+alu0) = (*(acc1+0));
  }
  if (alu23) {
    *(data0_128256+(alu0+4)) = (*(acc1+1));
  }
  if (alu23) {
    *(data0_128256+(alu0+8)) = (*(acc1+2));
  }
  if (alu23) {
    *(data0_128256+(alu0+12)) = (*(acc1+3));
  }
}