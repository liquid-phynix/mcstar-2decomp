#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <vector>
#include <string>
#include <functional>

#include "utils.hpp"
#include "bookkeeping.hpp"
#include "fft-fftw.hpp"
#include "array.hpp"

#ifdef SINGLEFLOAT
typedef float Float;
#else
typedef double Float;
#endif
typedef std::complex<Float> Complex;

//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = rand() / Float(RAND_MAX); }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi0; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi1; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi2; }
//void init_cmpl_array(const int& gi0, const int& gi1, const int& gi2, std::complex<Float>& v){ v = {Float(gi0), Float(gi1)}; }

typedef FFTW::Array<Float, ArrayDecomp::ArrayDecomp> Array;

void operator>>(Array& from, Array& to){
    if(not (from.realdec==to.realdec and from.cmpldec==to.cmpldec and from.dectype==to.dectype and from.acctype==to.acctype))
        ERROR("arrays cannot be globally transposed");
    if(from.is_real() and to.is_real())
        global_transposition(from.real_ptr(), from.dectype, to.real_ptr(), to.dectype, from.realdec.get_index());
    else if(from.is_cmpl() and to.is_cmpl())
        global_transposition(from.real_ptr(), from.dectype, to.real_ptr(), to.dectype, to.cmpldec.get_index());
    else ERROR("elem type must be the same for global transposition");
}

#include "dfft.hpp"

int main(int argc, char* argv[]){
    int3 gshape = (argc == 4) ? int3({atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}) : int3({80, 80, 80});

    Bookkeeping<Float> bk(gshape);

    Array arrA(gshape);
    Array arrB(arrA);

    FFTW::FFT<Float, Array> fft(arrA, arrB);

    arrB.as_cmpl();
    arrB.as_z();
    DFFT::r2c(fft, arrA, arrB);

    //TimeAcc tm;
    ////for(int it = 1; it <= 10; it++){
        //tm.start();
        //fft.r2c();
        //arr_cmpl.save("intermediate.bin");
        //fft.c2r();
        //tm.stop();
    ////}
    //arr_real.save("output.bin");

    //if(bk.rank == 0) tm.report_avg_ms("on average a round of fft took %f ms\n");
    //if(bk.rank == 0) std::cerr << "program terminating" << std::endl;

    return EXIT_SUCCESS;
}
