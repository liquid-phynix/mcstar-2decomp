.PHONY: all, clean

all: main_single

main_single: main_single.o interop.o
	mpif90 -lmpi_cxx main.o interop.o ../2decomp_fft_single/lib/lib2decomp_fft.a -o main

main_single.o: main.cpp
	mpic++ -DSINGLEFLOAT -c main.cpp -o main.o

interop.o: interop.f90
	mpif90 -I ../2decomp_fft_single/include -c interop.f90

clean:
	@(rm -f main_single main main_double main_single.o main.o |& > /dev/null; true)
	@(rm -f interop.o iop.mod |& > /dev/null; true)
