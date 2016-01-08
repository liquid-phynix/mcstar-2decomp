.PHONY: all, clean

all: main_single

main_single: main_single.o interop.o
	mpic++ -g -lOpenCL -L ../clFFT/build/library/ -lclFFT -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lgfortran -lmpi_cxx main_single.o interop.o ../2decomp_fft_single/lib/lib2decomp_fft.a -o $@

main_single.o: main.cpp
	mpic++ -g -Wno-deprecated-declarations -std=c++11 -I ../clFFT/src/include/ -DSINGLEFLOAT -c main.cpp -o $@

interop.o: interop.f95
	mpif90 -fbounds-check -g -std=f2003 -I ../2decomp_fft_single/include -c interop.f95 -o $@
	rm decomp_2d_interop.mod

clean:
	@(rm -f main_single main main_double main_single.o main.o |& > /dev/null; true)
	@(rm -f interop.o iop.mod |& > /dev/null; true)
