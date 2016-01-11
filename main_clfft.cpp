#include <iostream>
#include <complex>

#include "timing.hpp"

namespace DecompImpl {
    TimeAcc tm1;
    TimeAcc tm2;
    TimeAcc tm3;
    TimeAcc tm4;
}

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
    //DecompCLFFT::start_decomp_context(gshape, 1, GPU);
    DecompCLFFT::start_decomp_context(gshape, 0, CPU);

    DecompArray arrA(gshape);
    DecompArray arrB(arrA);

    arrA.as_x().as_real();
    arrB.as_z().as_cmpl();

    over<RT>(arrA, [](const int& gi0, const int& gi1, const int& gi2, RT& v){ v = (rand() / RT(RAND_MAX)); });

    DistributedFFT fft(arrA.decinfo);
    MASTER std::cout << "fft initialized" << std::endl;

    TimeAcc tm;
    for(int it = 1; it <= 10; it++){
        tm.start();
        fft.r2c(arrA, arrB);
        fft.c2r(arrB, arrA);
        tm.stop();
    }

    MASTER tm.report_avg_ms("one full %f ms\n");
    MASTER DecompImpl::tm1.report_avg_ms("fft > %f ms\n");
    MASTER DecompImpl::tm2.report_avg_ms("fft < %f ms\n");
    MASTER DecompImpl::tm3.report_avg_ms("trans > %f ms\n");
    MASTER DecompImpl::tm4.report_avg_ms("trans < %f ms\n");

    DecompCLFFT::end_decomp_context();
    MASTER std::cout << "terminating" << std::endl;
    return EXIT_SUCCESS;
}
