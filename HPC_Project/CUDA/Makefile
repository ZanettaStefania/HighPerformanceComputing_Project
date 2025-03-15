CC=nvc++
# CFLAGS=-qopenmp
TARGET=CUDA_mandelbrot
SRC=mandelbrot_CUDA_makefile.cu

# Resolution
RESOLUTION=1000 2000 4000

# Blocks
BLOCK=1 2 4 8 16 24 32 64 128 256 512 1024

all: $(TARGET)
# $(CC) $(CFLAG) -o $(TARGET) $(SRC)

$(TARGET): $(SRC)
	$(CC) -o $(TARGET) $(SRC)

run:
	@$(foreach RES, $(RESOLUTION), \
		$(foreach B, $(BLOCK), \
			echo Running with Res=$(RES) and BLOCKS=$(B); \
			echo makeout$(RES)_$(B); \
			./$(TARGET) outputs/makeout$(RES)_$(B).txt $(RES) $(B); \
		) \
	)

clean:
	rm -f $(TARGET)

.PHONY: all run clean