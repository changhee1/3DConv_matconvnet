// cudamx.cuh
//
// Author: Changhee Won (changhee.1.won@gmail.com)
//

#ifndef _CUDA_MX_CUH
#define _CUDA_MX_CUH

#define BLOCK2D 16
#define BLOCK1D 1024
#define BLOCK3D 8
#define WARP_SIZE 32

inline int NumGrid(int n, int block_size) {
  int v = n / block_size;
  return (n % block_size == 0) ? v : v+1;
}

#endif

