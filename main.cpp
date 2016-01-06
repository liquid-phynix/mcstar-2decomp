#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <vector>
#include <string>
#include <functional>

#include "decomp-fftw.hpp"
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

using namespace DecompWithFFTW;

int main(int argc, char* argv[]){
    int3 gshape = (argc == 4) ? int3({atoi(argv[1]), atoi(argv[2]), atoi(argv[3])}) : int3({80, 80, 80});

    Bookkeeping<Float> bk(gshape);

    DecompArray<Float> arrA(gshape);
    DecompArray<Float> arrB(arrA);
    DistributedFFT<Float> fft(arrA, arrB);

    arrA.over<Float>([](const int& gi0, const int& gi1, const int& gi2, Float& v){
        v = Float(rand() / Float(RAND_MAX)); });
    arrA.save("real.bin");
    arrB.as_cmpl().as_z();
    fft.r2c(arrA, arrB);
    arrB.save("cmpl.bin");
    fft.c2r(arrB, arrA);
    arrA.save("real_back.bin");

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

    return EXIT_SUCCESS;
}
