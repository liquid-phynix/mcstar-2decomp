#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <vector>
#include <string>
#include <functional>
#include <mpi.h>

#include "utils.hpp"
#include "array.hpp"
#include "bookkeeping.hpp"
#include "decomp.hpp"
#include "dfft.hpp"
#include "timing.hpp"

#ifdef SINGLEFLOAT
typedef float Float;
#else
typedef double Float;
#endif
typedef std::complex<Float> Complex;

void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi0; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi1; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi2; }
void init_cmpl_array(const int& gi0, const int& gi1, const int& gi2, std::complex<Float>& v){ v = {Float(gi0), Float(gi1)}; }

int main(int argc, char* argv[]){
    int3 real_shape;
    if(argc == 4){
        real_shape = {atoi(argv[1]), atoi(argv[2]), atoi(argv[2])};
    }
    else real_shape = {80, 80, 80};
    int3 cmpl_shape = to_hermitian(real_shape);
    Bookkeeping<Float> bk(real_shape);
    if(bk.rank == 0) std::cout << "complex shape = " << cmpl_shape << std::endl;

    DecompInfo decomp_real(real_shape);
    DecompInfo decomp_cmpl(cmpl_shape);
    //std::cout << decomp_real << std::endl;
    //std::cout << decomp_cmpl << std::endl;

    Array<Float> arr_real(decomp_real);
    Array<Complex> arr_cmpl(decomp_cmpl);
    arr_cmpl.set_z_pencil();

    arr_real.over(init_array);
    arr_cmpl.over(init_cmpl_array);

    DistributedFFT<typename FFTW::FFT<Float> > fft(decomp_real, decomp_cmpl, arr_real, arr_cmpl);

    TimeAcc tm;
    for(int it = 1; it <= 10; it++){
        tm.start();
        fft.r2c();
        fft.c2r();
        tm.stop();
    }

    if(bk.rank == 0) printf("on average a round of fft took %f ms\n");
    if(bk.rank == 0) std::cerr << "program terminating" << std::endl;

    return EXIT_SUCCESS;
}
