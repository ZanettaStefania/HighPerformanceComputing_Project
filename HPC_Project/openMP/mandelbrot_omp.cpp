// OMP

#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <omp.h>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X) //3
#define RATIO_Y (MAX_Y - MIN_Y) //2

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION) //3000
#define HEIGHT (RATIO_Y * RESOLUTION) //2000

#define STEP ((double)RATIO_X / WIDTH) //0.001

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv)
{
    // Set the number of threads to use in the parallelized region 
    const int n_threads = 512;
    omp_set_num_threads(n_threads);
    
    int *const image = new int[HEIGHT * WIDTH];  

    const auto start = chrono::steady_clock::now();

    cout << "Cores: " << n_threads << endl;
    cout << "Resolution: " << RESOLUTION << endl;


    // Parallelize this loop with openMP
    // #pragma omp parallel for schedule(static)
    // #pragma omp parallel for schedule(guided)
    // #pragma omp parallel for schedule(dynamic)

    // #pragma omp parallel for schedule(dynamic, 10417)
    // #pragma omp parallel for schedule(dynamic, 20833)
    // #pragma omp parallel for schedule(dynamic, 41666)
    #pragma omp parallel for
    for (int pos = 0; pos < HEIGHT * WIDTH; pos++)
    {
        image[pos] = 0;

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++) // Potentially the most intensive part 
        {
            // Operation on complex numbers -> intensive 
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
        }
    }
    
    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " milliseconds." << endl;

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
    return 0;
}