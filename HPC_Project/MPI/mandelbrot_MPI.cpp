#include <iostream>
#include <fstream>
#include <complex>
#include <chrono>
#include <mpi.h>

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 4000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations

using namespace std;

int main(int argc, char **argv)
{
    int *const image = new int[HEIGHT * WIDTH];
    int rec_id = 0;
    double start, end;
    int remainder; 

    MPI_Init(NULL, NULL);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    remainder = (HEIGHT*WIDTH) % world_size;


    int local_size = (HEIGHT*WIDTH) / world_size;
    int local_size_res;
    int* local_image = nullptr;

    int pos_start, pos_end;
    if(rank == world_size - 1 && ((rank*local_size+local_size-1) < (HEIGHT*WIDTH) - 1)){

        local_size_res = (HEIGHT*WIDTH) - rank * local_size;
        local_image = new int[local_size_res];
        pos_start = rank * local_size;
        pos_end = (HEIGHT*WIDTH) - 1;
        local_size = local_size_res;
    }
    else{

        local_image = new int[local_size];
        pos_start = rank * local_size;      
        pos_end = pos_start + local_size - 1;
    }
    
    start = MPI_Wtime();
    

    int j=0;
    for (int pos = pos_start; pos <= pos_end; pos++)
    {
        local_image[j] = 0;

        const int row = pos / WIDTH;
        const int col = pos % WIDTH;
        const complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

        // z = z^2 + c
        complex<double> z(0, 0);
        for (int i = 1; i <= ITERATIONS; i++)
        {
            z = pow(z, 2) + c;

            // If it is convergent
            if (abs(z) >= 2)
            {
                local_image[j] = i;
                break;
            }
        }
        j++;
    }


    if(remainder==0){
        MPI_Gather(local_image, local_size, MPI_INT, image, local_size, MPI_INT, rec_id, MPI_COMM_WORLD);
        end = MPI_Wtime();
    }
    else{
        int* counts; 
        int* displacements;
        if(rank==rec_id){
            counts = new int[world_size];
            displacements = new int[world_size];

            for(int i = 0; i < world_size - 1; i++){
                counts[i] = local_size;
                displacements[i] = i * local_size;
            }

            counts[world_size - 1] = ((HEIGHT*WIDTH)) - ((world_size - 1) * local_size);
            displacements[world_size - 1] = (world_size - 1) * local_size;

        }
        MPI_Gatherv(local_image, local_size, MPI_INT, image, counts, displacements, MPI_INT, rec_id, MPI_COMM_WORLD);
        end = MPI_Wtime();
    }

    

    // Write the result to a file
    ofstream matrix_out;
    if(rank == rec_id){
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

        printf("Processes: %d\n", world_size);
        printf("Elapsed time (s) = %.2lf\n", end - start);
    }
    MPI_Finalize();
    return 0;
}