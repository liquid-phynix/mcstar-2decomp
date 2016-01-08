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
#include "decomp-clfft.hpp"
using namespace DecompCLFFT;

int main(int argc, char* argv[]){
    int3 gshape = (argc == 4) ? int3({atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}) : int3({80, 80, 80});
    DecompCLFFT::start_decomp_context(gshape, 1, GPU);
    //DecompCLFFT::start_decomp_context(gshape, 0, CPU);

    DecompArray arrA(gshape);
    DecompArray arrB(arrA);

    arrA.as_cmpl() >> arrB.as_cmpl().as_y();

    over<RT>(arrA.as_real(), [](const int& gi0, const int& gi1, const int& gi2, Decomp::RT& v){ v = (rand() / RT(RAND_MAX)); });

    DistributedFFT fft(arrA.decinfo);
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

    DecompCLFFT::end_decomp_context();
    std::cout << "terminating" << std::endl;
    return EXIT_SUCCESS;
}
