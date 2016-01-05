#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <vector>
#include <string>
#include <functional>

#include "decomp.hpp"
#include "utils.hpp"

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

#define MASTER if(bk.get_rank() == 0)

using namespace Decomp;

int main(int argc, char* argv[]){
    int3 gshape = (argc == 4) ? int3({atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}) : int3({80, 80, 80});

    Bookkeeping<Float> bk(gshape);

    DecompArray<Float> arrA(gshape);
    DecompArray<Float> arrB(arrA);
    arrA >> arrB.as_y();
    //Array arrB(arrA);

    DistributedFFT<Float> fft;
    fft.r2c(arrA, arrB.as_cmpl().as_z());
    fft.c2r(arrB, arrA);

    //TimeAcc tm;
    //for(int it = 1; it <= 10; it++){
        //tm.start();
        //DFFT::r2c(fft, arrA, arrB);
        //DFFT::c2r(fft, arrB, arrA);
        //tm.stop();
    //}
    ////arr_real.save("output.bin");

    //MASTER tm.report_avg_ms("on average a round of fft took %f ms\n");
    ////if(bk.rank == 0) std::cerr << "program terminating" << std::endl;

    return EXIT_SUCCESS;
}
