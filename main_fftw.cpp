#include <iostream>
#include <complex>

namespace Decomp {
#ifdef SINGLEFLOAT
    typedef float RT;
#else
    typedef double RT;
#endif
    typedef std::complex<RT> CT;
}
#include "decomp-fftw.hpp"
using namespace DecompFFTW;

int main(int argc, char* argv[]){
    int3 gshape = (argc == 4) ? int3({atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}) : int3({80, 80, 80});
    start_decomp_context(gshape);

    DecompArray arrA(gshape);
    DecompArray arrB(arrA);

    arrA.as_cmpl() >> arrB.as_cmpl().as_y();

    over<RT>(arrA.as_real(), [](const int& gi0, const int& gi1, const int& gi2, Decomp::RT& v){ v = (rand() / RT(RAND_MAX)); });

    DistributedFFT fft(arrA, arrB);
    std::cout << "fft initialized" << std::endl;

    arrB.as_z();

    TimeAcc tm;
    for(int it = 1; it <= 10; it++){
        tm.start();
        fft.r2c(arrA, arrB);
        fft.c2r(arrB, arrA);
        tm.stop();
    }

    MASTER tm.report_avg_ms("on average a round of fft took %f ms\n");

    end_decomp_context();
    std::cout << "terminating" << std::endl;
    return EXIT_SUCCESS;
}