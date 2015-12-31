.PHONY: all, clean

CLFFTBASE=/home/mcstar/src/clFFT

all: main_single

main_single: main_single.o interop.o
	mpic++ -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lgfortran -lOpenCL -L$(CLFFTBASE)/build/library -lclFFT -lmpi_cxx main.o interop.o ../2decomp_fft_single/lib/lib2decomp_fft.a -o $@

main_single.o: main.cpp
	mpic++ -Wno-deprecated-declarations -std=c++11 -DSINGLEFLOAT -I$(CLFFTBASE)/src/include -c main.cpp -o main.o

interop.o: interop.f95
	mpif90 -std=f2003 -I ../2decomp_fft_single/include -c interop.f95 -o $@

clean:
	@(rm -f main_single main main_double main_single.o main.o |& > /dev/null; true)
	@(rm -f interop.o iop.mod |& > /dev/null; true)
