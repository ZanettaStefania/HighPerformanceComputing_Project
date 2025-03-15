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
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Default values
#define DEFAULT_RESOLUTION 1000
#define DEFAULT_WIDTH (RATIO_X * DEFAULT_RESOLUTION)
#define DEFAULT_HEIGHT (RATIO_Y * DEFAULT_RESOLUTION)
#define DEFAULT_STEP ((double)RATIO_X / DEFAULT_WIDTH)
#define DEFAULT_THREADS 4

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv)
{
    int resolution = DEFAULT_RESOLUTION;
    int n_threads = DEFAULT_THREADS;

    if (argc > 1)
    {
        resolution = atoi(argv[1]);
        if (resolution <= 0)
        {
            cout << "Invalid resolution. Defaulting to " << DEFAULT_RESOLUTION << endl;
            resolution = DEFAULT_RESOLUTION;
        }
    }

    if (argc > 2)
    {
        n_threads = atoi(argv[2]);
        if (n_threads <= 0)
        {
            cout << "Invalid number of threads. Defaulting to " << DEFAULT_THREADS << endl;
            n_threads = DEFAULT_THREADS;
        }
    }

    const double step = (double)RATIO_X / (RATIO_X * resolution);
    const int width = RATIO_X * resolution;
    const int height = RATIO_Y * resolution;
    int *const image = new int[height * width];

    const auto start = chrono::steady_clock::now();

    cout << "Resolution: " << resolution << endl;
    cout << "Threads: " << n_threads << endl;

    // Set the number of threads to use in this region
    omp_set_num_threads(n_threads);

    // Parallelize this loop with OpenMP
    // #pragma omp parallel for schedule(static)
    // #pragma omp parallel for schedule(guided)
    // #pragma omp parallel for schedule(dynamic)

    // #pragma omp parallel for schedule(dynamic, 10417)
    // #pragma omp parallel for schedule(dynamic, 20833)
    // #pragma omp parallel for schedule(dynamic, 41666)
    #pragma omp parallel for
    for (int pos = 0; pos < height * width; pos++)
    {
        image[pos] = 0;

        const int row = pos / width;
        const int col = pos % width;
        const complex<double> c(col * step + MIN_X, row * step + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++) // Potentially the most intensive part
        {
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

    if (argc < 4)
    {
        cout << "Please specify the output file as a parameter." << endl;
        return -1;
    }

    matrix_out.open(argv[3], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        return -2;
    }

    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            matrix_out << image[row * width + col];

            if (col < width - 1)
                matrix_out << ',';
        }
        if (row < height - 1)
            matrix_out << endl;
    }
    matrix_out.close();

    delete[] image;
    return 0;
}
