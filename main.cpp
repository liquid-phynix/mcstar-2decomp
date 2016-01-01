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
#include "fft-fftw.hpp"
#include "decomp.hpp"

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

template<class FFT>
class DistributedFFT : protected FFT {
public:
    DistributedFFT(DecompInfo rd, DecompInfo cd, Array<typename FFT::RT>& ra, Array<typename FFT::CT>& ca):
    FFT(rd, cd, ra.ptr, ca.ptr){
    }
};

int main(int argc, char* argv[]){
    //int3 real_shape = {80, 80, 80};
    int3 real_shape = {512, 512, 512};
    int3 cmpl_shape = to_hermitian(real_shape);
    Bookkeeping<Float> bk(real_shape);
    if(bk.rank == 0) std::cout << "complex shape = " << cmpl_shape << std::endl;

    DecompInfo decomp_real(real_shape);
    DecompInfo decomp_cmpl(cmpl_shape);
    //std::cout << decomp_real << std::endl;
    //std::cout << decomp_cmpl << std::endl;

    Array<Float> arr_real(decomp_real);
    Array<Complex> arr_cmpl(decomp_cmpl);
    //Array<Complex> arr3(decomp_cmpl);


    //typedef  BaseClass;
    //DistributedFFT<Float, BaseClass> fft(decomp_real, decomp_cmpl, arr_real, arr_cmpl);
    DistributedFFT<typename FFTW::FFT<Float> > fft(decomp_real, decomp_cmpl, arr_real, arr_cmpl);

    //arr1.over(init_array);
    //arr2.over(init_cmpl_array);
    //arr3.over(init_cmpl_array);

    ////arr1.save("real.bin");
    //arr2.save("cmpl1.bin");


    //for(int i = 0; i < 10; i++){
    //arr2.set_x_pencil();
    //arr3.set_y_pencil();
    //arr2.transpose_into(arr3);
    //arr2.set_z_pencil();
    //arr3.transpose_into(arr2);

    //arr2.transpose_into(arr3);
    //arr2.set_x_pencil();
    //arr3.transpose_into(arr2);
    //}
    //for(int i = 0; i < 100; i++){
        //arr2.transpose_into(arr3);
    //}

    //arr3.set_x_pencil();
    //arr2.set_y_pencil();
    //arr2.transpose_into(arr3);

    //arr3.transpose_into(arr2);

    //arr2.set_z_pencil();
    //arr3.transpose_into(arr2);

    //arr2.save("cmpl2.bin");


     if(bk.rank == 0) std::cerr << "program terminating" << std::endl;

    return EXIT_SUCCESS;
}
