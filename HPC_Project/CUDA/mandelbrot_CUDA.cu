#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <cmath>
#include "cuda_runtime.h"

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;



__global__ void mandelbrotGPUfunction(int *image, double step, double minX, double minY, int width, int height, int iterations)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos < width * height) {

        image[pos] = 0;
        const int row = pos / width;
        const int col = pos % width;

        const complex<double> c(col * step + minX, row * step + minY);

        complex<double> z(0, 0);

        for (int i = 1; i <= iterations; i++){
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
            if(i == iterations){
                image[pos]= 0;
            }
        }
    }
}



int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];

    // Timer
    cudaEvent_t  start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    // Allocation
    int *d_image;
    cudaMalloc(&d_image, sizeof(int) * WIDTH * HEIGHT);
    
    // Threads
    dim3 block(1024); // threads per block 
    dim3 grid((WIDTH * HEIGHT + block.x - 1) / block.x);
    
    // Start timer
    cudaEventRecord(start);

    // Function call
    mandelbrotGPUfunction<<<grid, block>>>(d_image, STEP, MIN_X, MIN_Y, WIDTH, HEIGHT, ITERATIONS);
    cudaDeviceSynchronize();

    // Copy Results
    cudaMemcpy(image, d_image, sizeof(int) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute Time
    float elapsed_time = 0.0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cout << "Time elapsed: " <<  elapsed_time << " milliseconds." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    cudaFree(d_image);
    return 0;
}