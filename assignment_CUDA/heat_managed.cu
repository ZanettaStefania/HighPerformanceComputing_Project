#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

/*
 * `step_kernel_mod` is currently a direct copy of the CPU reference solution
 * `step_kernel_ref` below. Accelerate it to run as a CUDA kernel.
 */

__global__ void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  int i = blockIdx.x * blockDim.x + threadIdx.x; //global id of thread
  int stride = blockDim.x * gridDim.x;


  // loop over all points in domain (except boundary)
  for ( int idx=i; idx < (nj-2)*(ni-2); idx+=stride ) {
      // find indices into linear memory
      // for central point and neighbours
      int i = idx % (ni-2)+1;
      int j = idx / (ni-2)+1;

      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  int istep;
  int nstep = 200; // number of time steps

  // Specify our 2D dimensions
  const int ni = 10000;
  const int nj = 10000;
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp_tmp, *temp1_mod, *temp2_mod;
  const int size = ni * nj * sizeof(float);

  dim3 threads(2); 
  dim3 dimblock(((ni-ni/2)*(nj-nj/2)+ threads.x-1)/threads.x);

  
  cudaEvent_t start, stop, start_ref, stop_ref;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&start_ref);
  cudaEventCreate(&stop_ref);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);

  cudaMallocManaged(&temp1_mod, size);
  cudaMallocManaged(&temp2_mod, size);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1_mod[i] = temp2_mod[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  
  // ---------- CPU ------------
  cudaEventRecord(start_ref, 0);
  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }
  cudaEventRecord(stop_ref, 0);
  
  
  // ------------ GPU ----------------
  cudaEventRecord(start, 0);
  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {
    step_kernel_mod<<<dimblock, threads>>>(ni, nj, tfac, temp1_mod, temp2_mod);
    cudaDeviceSynchronize();
    // swap the temperature pointers
    temp_tmp = temp1_mod;
    temp1_mod = temp2_mod;
    temp2_mod= temp_tmp;
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);


  float elapsed = 0.0;
  float elapsed_ref = 0.0; 
  cudaEventElapsedTime(&elapsed_ref, start_ref, stop_ref);
  printf("Elapsed CPU: %.3f ms \n", elapsed_ref);

  cudaEventElapsedTime(&elapsed, start, stop);
  printf("Elapsed GPU: %.3f ms \n", elapsed);

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1_mod[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1_mod[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.8f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.8f is within acceptable bounds.\n", maxError);
  
  cudaFree( temp1_mod );
  cudaFree( temp2_mod );

  free( temp1_ref );
  free( temp2_ref );

  return 0;
}