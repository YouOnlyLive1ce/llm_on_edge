#include <stdint.h>
#include <stdio.h>

// todo: visualizer of grid(block_num,thread_num) kernel performance 

__global__ void gemm_row_col(float* a, float* b, float* res, int width,int common_dim) {
    // one thread per one dot product
    int x=threadIdx.x;//0=+blockDim.x*blockIdx.x; // [0..width]
    int y=threadIdx.y;//0=+blockDim.y*blockIdx.y; // [0..height]

    if (x<width){ // do not access dummy threads
        for (int i=0;i<common_dim;i++)
            res[x+y*blockDim.y]+=a[x+blockDim.y*i]*b[i*blockDim.x+y];
    }
}

// __global__ void transpose(float* x){
    
// }
// __global__ void gemm_row_row(float* a, float* b, float* res) {
    
// }

//__global__ void gemm_parallel_dot(float* a, float* b, float* res) {
    
//}

int main() {
  int width = 4, height = 4, depth = 1;
  
  // Allocate arrays for X and Y on the CPU. This memory is only usable on the CPU
  float* cpu_x = (float*)malloc(sizeof(float) * width*height);
  float* cpu_y = (float*)malloc(sizeof(float) * width*height);
  float* cpu_res = (float*)malloc(sizeof(float) * width*height);
  for(int i=0; i<width*height; i++) {
    cpu_x[i] = (float)i+1.1f;
    cpu_y[i] = (float)i*2.1f;
  }

  // The gpu_x and gpu_y pointers will only be usable on the GPU (which uses separate memory)
  float *gpu_x, *gpu_y, *gpu_res;
  if(cudaMalloc(&gpu_x, sizeof(float) *width*height*depth) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate X array on GPU\n");
    exit(2);
  }
  if(cudaMalloc(&gpu_y, sizeof(float) *width*height*depth) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate Y array on GPU\n");
    exit(2);
  }
  if(cudaMalloc(&gpu_res, sizeof(float) *width*height*depth) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate res array on GPU\n");
    exit(2);
  }

  // Copy the cpu's x array to the gpu with cudaMemcpy
  if(cudaMemcpy(gpu_x, cpu_x, sizeof(float) * width*height, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy X to the GPU\n");
  }
  if(cudaMemcpy(gpu_y, cpu_y, sizeof(float) * width*height, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y to the GPU\n");
  }

  // Calculate the number of blocks to run, rounding up to include all threads
  dim3 block_dim=(width,height);
  dim3 grid_dim=(1,1);
  gemm_row_col<<<grid_dim, block_dim>>>(gpu_x, gpu_y,gpu_res,height,width);
  // Wait for the kernel to finish
  if(cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }

  // Copy the y array back from the gpu to the cpu
  if(cudaMemcpy(cpu_res, gpu_res, sizeof(float) * width*height*depth, cudaMemcpyDeviceToHost) != cudaSuccess) {
    fprintf(stderr, "Failed to copy Y from the GPU\n");
  }
  for(int i=0; i<width*height*depth; i++) {
    printf("%d: %f\n", i, cpu_res[i]);
  }

  cudaFree(gpu_x);
  cudaFree(gpu_y);
  free(cpu_x);
  free(cpu_y);

  return 0;
}
