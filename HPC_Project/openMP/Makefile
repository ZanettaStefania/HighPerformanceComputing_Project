CC = icc
CFLAGS = -diag-disable=10441 -Ofast -qopenmp

all: mandelbrot_omp_mk

mandelbrot_omp_mk: mandelbrot_omp_makefile.cpp
	$(CC) $(CFLAGS) -o mandelbrot_omp_mk mandelbrot_omp_makefile.cpp

clean:
	rm -f mandelbrot_omp_mk

run_4000_512:
	@echo "Compiling program with 4000 resolution and number of thread 512..."
	./mandelbrot_omp_mk 4000 512 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_512:
	@echo "Compiling program with 2000 resolution and number of thread 512..."
	./mandelbrot_omp_mk 2000 512 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_512:
	@echo "Compiling program with 1000 resolution and number of thread 512..."
	./mandelbrot_omp_mk 1000 512 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_256:
	@echo "Compiling program with 4000 resolution and number of thread 256..."
	./mandelbrot_omp_mk 4000 256 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_256:
	@echo "Compiling program with 2000 resolution and number of thread 256..."
	./mandelbrot_omp_mk 2000 256 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_256:
	@echo "Compiling program with 1000 resolution and number of thread 256..."
	./mandelbrot_omp_mk 1000 256 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_128:
	@echo "Compiling program with 4000 resolution and number of thread 128..."
	./mandelbrot_omp_mk 4000 128 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_128:
	@echo "Compiling program with 2000 resolution and number of thread 128..."
	./mandelbrot_omp_mk 2000 128 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_128:
	@echo "Compiling program with 1000 resolution and number of thread 128..."
	./mandelbrot_omp_mk 1000 128 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_64:
	@echo "Compiling program with 4000 resolution and number of thread 64..."
	./mandelbrot_omp_mk 4000 64 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_64:
	@echo "Compiling program with 2000 resolution and number of thread 64..."
	./mandelbrot_omp_mk 2000 64 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_64:
	@echo "Compiling program with 1000 resolution and number of thread 64..."
	./mandelbrot_omp_mk 1000 64 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_32:
	@echo "Compiling program with 4000 resolution and number of thread 32..."
	./mandelbrot_omp_mk 4000 32 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_32:
	@echo "Compiling program with 2000 resolution and number of thread 32..."
	./mandelbrot_omp_mk 2000 32 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_32:
	@echo "Compiling program with 1000 resolution and number of thread 32..."
	./mandelbrot_omp_mk 1000 32 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_24:
	@echo "Compiling program with 4000 resolution and number of thread 24..."
	./mandelbrot_omp_mk 4000 24 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_24:
	@echo "Compiling program with 2000 resolution and number of thread 24..."
	./mandelbrot_omp_mk 2000 24 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_24:
	@echo "Compiling program with 1000 resolution and number of thread 24..."
	./mandelbrot_omp_mk 1000 24 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_16:
	@echo "Compiling program with 4000 resolution and number of thread 16..."
	./mandelbrot_omp_mk 4000 16 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_16:
	@echo "Compiling program with 2000 resolution and number of thread 16..."
	./mandelbrot_omp_mk 2000 16 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_16:
	@echo "Compiling program with 1000 resolution and number of thread 16..."
	./mandelbrot_omp_mk 1000 16 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_8:
	@echo "Compiling program with 4000 resolution and number of thread 8..."
	./mandelbrot_omp_mk 4000 8 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_8:
	@echo "Compiling program with 2000 resolution and number of thread 8..."
	./mandelbrot_omp_mk 2000 8 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_8:
	@echo "Compiling program with 1000 resolution and number of thread 8..."
	./mandelbrot_omp_mk 1000 8 output.txt
	@echo "/--------------------------------------------------------------/"

run_4000_4:
	@echo "Compiling program with 4000 resolution and number of thread 4..."
	./mandelbrot_omp_mk 4000 4 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_4:
	@echo "Compiling program with 2000 resolution and number of thread 4..."
	./mandelbrot_omp_mk 2000 4 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_4:
	@echo "Compiling program with 1000 resolution and number of thread 4..."
	./mandelbrot_omp_mk 1000 4 output_custom_res.txt
	@echo "/--------------------------------------------------------------/"

run_4000_2:
	@echo "Compiling program with 4000 resolution and number of thread 2..."
	./mandelbrot_omp_mk 4000 2 output.txt
	@echo "/--------------------------------------------------------------/"

run_2000_2:
	@echo "Compiling program with 2000 resolution and number of thread 2..."
	./mandelbrot_omp_mk 2000 2 output.txt
	@echo "/--------------------------------------------------------------/"

run_1000_2:
	@echo "Compiling program with 1000 resolution and number of thread 2..."
	./mandelbrot_omp_mk 1000 2 output_custom_res.txt
	@echo "/--------------------------------------------------------------/"

run_all: run_4000_512 run_2000_512 run_1000_512 \
         run_4000_256 run_2000_256 run_1000_256 \
         run_4000_128 run_2000_128 run_1000_128 \
         run_4000_64 run_2000_64 run_1000_64 \
         run_4000_32 run_2000_32 run_1000_32 \
         run_4000_24 run_2000_24 run_1000_24 \
         run_4000_16 run_2000_16 run_1000_16 \
         run_4000_8 run_2000_8 run_1000_8 \
         run_4000_4 run_2000_4 run_1000_4 \
         run_4000_2 run_2000_2 run_1000_2