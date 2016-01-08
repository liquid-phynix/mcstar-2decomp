#include <iostream>
#include <complex>
#include <functional>

namespace Decomp {
#ifdef SINGLEFLOAT
    typedef float RT;
#else
    typedef double RT;
#endif
    typedef std::complex<RT> CT;
}

#include "new-decomp.hpp"
using namespace Decomp;

int main(int argc, char* argv[]){
    int3 gshape = (argc == 4) ? int3({atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}) : int3({80, 80, 80});
    start_decomp_context(gshape);

    DecompArray arrA(gshape);
    DecompArray arrB(arrA);

    arrA.save("real.bin");

    arrA.as_cmpl() >> arrB.as_cmpl().as_y();


    over<RT>(arrA.as_real(), [](const int& gi0, const int& gi1, const int& gi2, Decomp::RT& v){ v = (rand() / RT(RAND_MAX)); });


    DistributedFFT fft;

    arrB.as_z();
    std::cout << "arrA=" << arrA << std::endl;
    std::cout << "arrB=" << arrB << std::endl;

    fft.r2c(arrA, arrB);
    fft.c2r(arrB, arrA);

    //arrA.save("real.bin");
    //arrB.as_cmpl().as_z();
    //fft.r2c(arrA, arrB);
    //arrB.save("cmpl.bin");
    //fft.c2r(arrB, arrA);
    //arrA.save("real_back.bin");

    //arrA.over<Complex>([](const int& gi0, const int& gi1, const int& gi2, Complex& v){
        //v = {Float(rand() / Float(RAND_MAX)), Float(rand() / Float(RAND_MAX))}; });


    //arrA.as_cmpl().as_z();
    //arrB.as_cmpl().as_z();
    //arrA.over<Complex>([](const int& gi0, const int& gi1, const int& gi2, Complex& v){ v = {Float(rand() / Float(RAND_MAX)),Float(rand() / Float(RAND_MAX))}; });
    //arrA.save("cmpl.bin");

    ////arrB.as_cmpl().over<Complex>([](const int& gi0, const int& gi1, const int& gi2, Complex& v){ v = {Float(gi0), Float(gi1)}; });
    ////arrB.save("cmpl.bin");

    ////fft.r2c(arrA, arrB.as_cmpl());
    //fft.forward(arrA, arrB);
    //arrB.save("cmpl_zf.bin");
    //fft.backward(arrB, arrA);
    ////std::cerr << "arrA: " << arrA << std::endl;
    //arrA.save("cmpl_back.bin");
    //arrA >> arrB.as_y();
    //arrB >> arrA;

    //arrB.as_cmpl().as_z();
    //arrB.save("cmpl.bin");
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
