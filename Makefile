.PHONY: all, clean

all: main

main: main.o interop.o
	mpif90 -lmpi_cxx main.o interop.o ../2decomp_fft_single/lib/lib2decomp_fft.a -o main

main.o: main.cpp
	mpic++ -c main.cpp -o main.o

interop.o: interop.f90
	mpif90 -I ../2decomp_fft_single/include -c interop.f90

clean:
	-rm main main.o
	-rm interop.o interop.mod
