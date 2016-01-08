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
    DecompCLFFT::start_decomp_context(gshape);

    DecompArray arrA(gshape);
    DecompArray arrB(arrA);

    //arrA.save("real.bin");

    //arrA.as_cmpl() >> arrB.as_cmpl().as_y();


    //over<RT>(arrA.as_real(), [](const int& gi0, const int& gi1, const int& gi2, Decomp::RT& v){ v = (rand() / RT(RAND_MAX)); });

    //DistributedFFT fft(arrA.decinfo);

    //arrB.as_z();
    //std::cout << "arrA=" << arrA << std::endl;
    //std::cout << "arrB=" << arrB << std::endl;

    //fft.r2c(arrA, arrB);
    //fft.c2r(arrB, arrA);


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

    end_decomp_context();
    return EXIT_SUCCESS;
}
