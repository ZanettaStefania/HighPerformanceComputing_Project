#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>

// Ranges of the set
// Define the rectangular region in the complex plane where the Mandelbrot set will be computed
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image aspect ratio
#define RATIO_X (MAX_X - MIN_X) // 3
#define RATIO_Y (MAX_Y - MIN_Y) // 2

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION) // 3000
#define HEIGHT (RATIO_Y * RESOLUTION) // 2000

#define STEP ((double)RATIO_X / WIDTH) // 0.001

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv)
{
    // Dinamically allocate memory for the image
    // The value stored in each element will represent the iteration count at which the sequence for the 
    // corresponding point diverged (or 0 if it did not diverge within the maximum number of iterations).
    int *const image = new int[HEIGHT * WIDTH];
    const auto start = chrono::steady_clock::now();

    // The code iterates over each pixel in the image, and for each pixel, it computes the corresponding complex number c.
    for (int pos = 0; pos < HEIGHT * WIDTH; pos++)
    {
        image[pos] = 0;

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        // For each complex number c, the code initializes z to 0+0i and iterates the equation z = z^2 + c 
        // until the magnitude of z exceeds 2 or the maximum number of iterations is reached.
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is divergent
            // Check if the magnitude of z exceeds 2. If it does, the code stores the iteration count
            // in the corresponding element of the image and breaks the loop.
            if (abs(z) >= 2)
            {
                image[pos] = i;
                break;
            }
            // If the sequence does not diverge within the maximum number of iterations, 
            // the pixel value remains 0, which might be mapped to a specific color representing the Mandelbrot set.
            // So the Mandelbrot set i
        }
    }
    // The Mandelbrot set is represented by the pixels for which the iterative process z = z^2 + c does not diverge,
    // meaning that the sequence remains bounded.

    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::seconds>(end - start).count()
         << " seconds." << endl;

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