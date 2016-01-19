include host.mk

.PHONY: all, clean

all: fftw_single clfft_single clfft_single_uhp

clfft_single_uhp.o: main_clfft_uhp.cpp decomp.hpp decomp-clfft-uhp.hpp
	$(CXX) $(CFLAGS) $(CLINC) -I$(CLFFTINC) -Wno-deprecated-declarations -std=c++11 -DSINGLEFLOAT -c main_clfft_uhp.cpp -o $@
clfft_single_uhp: clfft_single_uhp.o interop.o
	$(CXX) $(LFLAGS) -L$(CLFFTLIB) -lOpenCL -lclFFT clfft_single_uhp.o interop.o $(LIB2DECOMPS) -o $@

clfft_single.o: main_clfft.cpp decomp.hpp decomp-clfft.hpp
	$(CXX) $(CFLAGS) $(CLINC) -I$(CLFFTINC) -Wno-deprecated-declarations -std=c++11 -DSINGLEFLOAT -c main_clfft.cpp -o $@
clfft_single: clfft_single.o interop.o
	$(CXX) $(LFLAGS) -L$(CLFFTLIB) -lOpenCL -lclFFT clfft_single.o interop.o $(LIB2DECOMPS) -o $@

fftw_single.o: main_fftw.cpp decomp.hpp decomp-fftw.hpp
	$(CXX) $(CFLAGS) -std=c++11 -DSINGLEFLOAT -c main_fftw.cpp -o $@
fftw_single: fftw_single.o interop.o
	$(CXX) $(LFLAGS) -lm -lfftw3f fftw_single.o interop.o $(LIB2DECOMPS) -o $@

interop.o: interop.f95
	$(FC) -fbounds-check -g -std=f2003 -I$(INC2DECOMPS) -c interop.f95 -o $@
	rm decomp_2d_interop.mod

clean:
	@(rm -f fftw_single fftw_single.o clfft_single clfft_single.o interop.o)
