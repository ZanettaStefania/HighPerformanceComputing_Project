#include <stdio.h>
#include <time.h>
#include <mpi.h>
               
#define PI25DT 3.141592653589793238462643
#define INTERVALS 100000000000

int main(int argc, char **argv)
{
        
    double x, dx, f, sum, pi;
    double time1, time2;
    double start_hot, end_hot;
    long int i, intervals = INTERVALS;
    
    MPI_Init(NULL, NULL); 
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    time1 = MPI_Wtime();

    long int local_interval = INTERVALS / size;
    double local_sum = 0.0;
    double global_sum = 0.0;
    long int start, end;

    if( rank == size-1 ){
        int remaind = INTERVALS % size;
        start = rank * local_interval + 1;
        end = (rank + 1) * local_interval + remaind;
    }
    else{
        start = rank * local_interval + 1;
        end = (rank + 1) * local_interval;
    }

    sum = 0.0;
    dx = 1.0 / (double) intervals;
    
    start_hot = MPI_Wtime();
    for (i = start; i <= end; i++) {
        x = dx * ((double) (i - 0.5));
        f = 4.0 / (1.0 + x*x);
        local_sum += f;
    }
    end_hot = MPI_Wtime();

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0 ){
        pi = dx * global_sum;
        
        time2 = MPI_Wtime();

        printf("Computed PI %.24f\n", pi);
        printf("The true PI %.24f\n\n", PI25DT);
        
        double relative_error = fabs((PI25DT - pi) / PI25DT);
        double similarity_percentage = (1 - relative_error) * 100;
        printf("Similarity Percentage: %.12f%%\n", similarity_percentage); 

        printf("Elapsed time TOTAL (s) = %.2lf\n", time2 - time1);
        printf("Elapsed time HOTSPOT (s) = %.2lf\n", end_hot - start_hot);
    }
    MPI_Finalize();
    return 0;
}