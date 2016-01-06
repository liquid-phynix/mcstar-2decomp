#pragma once
#include <fftw3.h>
#include <complex>
#include "decomp.hpp"

namespace DecompWithFFTWImpl {

#ifdef SINGLEFLOAT
#define FPREF(a) fftwf_##a
#else
#define FPREF(a) fftw_##a
#endif

    //using namespace DecompImpl;
    using DecompImpl::DecompInfo;

    template <class N>
    class FFTWDecompArrayBase : N {
        using typename N::RT;
        using typename N::CT;
    protected:
        CT* ptr;
        size_t alloc_bytes;
    public:
        FFTWDecompArrayBase() = delete;
        FFTWDecompArrayBase(DecompInfo cd){
            size_t alloc_len = std::max(std::max(cd.xsize.prod(), cd.ysize.prod()), cd.zsize.prod());
            alloc_bytes = sizeof(CT) * alloc_len;
            ptr = (CT*)malloc(alloc_bytes);
            memset(ptr, 0, alloc_bytes);
            //ptr = (CT*)FPREF(malloc)(alloc_bytes);
            int lock = mlock(ptr, alloc_bytes);
            if(lock) fprintf(stderr, "memory region cannot be pinned\n");
        }
        ~FFTWDecompArrayBase(){ FPREF(free)(ptr); }
        //~FFTWDecompArrayBase(){ free(ptr); }
        RT* real_ptr(){ return reinterpret_cast<RT*>(ptr); }
        CT* cmpl_ptr(){ return reinterpret_cast<CT*>(ptr); }
        FPREF(complex)* fftw_cmpl_ptr(){ return reinterpret_cast<FPREF(complex)*>(ptr); }
    };

    template <typename F>
    using DecompArray = DecompImpl::DecompArray<F, FFTWDecompArrayBase>;

    template <typename F>
    class FFTW : public DecompImpl::DistributedFFTBase<F, FFTW<F>> {
        FPREF(plan) plan_x_r2c, plan_x_c2r, plan_y_forw, plan_y_back, plan_z_forw, plan_z_back;
    public:
        FFTW() = delete;
        FFTW(const FFTW<F>&) = delete;
        FFTW(DecompArray<F>& scratchA, DecompArray<F>& scratchB){
            if(sizeof(FPREF(complex)) != 2 * sizeof(F)) ERROR("main() and fftw float types differ");
            FPREF(iodim) transform{}, repeat[3]{};
            int transform_rank = 1; // transform dimension
            int repeat_rank = 2;    // 2D repetition
            int real_array_size[3]{}; int array_size[3]{};

            // plan for x-directional decomposition
            // xsize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            real_array_size[0] = scratchA.realdec.xsize.z;
            real_array_size[1] = scratchA.realdec.xsize.y;
            real_array_size[2] = scratchA.realdec.xsize.x;
            array_size[0]      = scratchA.cmpldec.xsize.z;
            array_size[1]      = scratchA.cmpldec.xsize.y;
            array_size[2]      = scratchA.cmpldec.xsize.x;

            transform.n = real_array_size[2]; // transform length
            transform.is = transform.os = 1;
            repeat[0].n = real_array_size[0]; repeat[0].is = real_array_size[1] * real_array_size[2]; repeat[0].os = array_size[1] * array_size[2];
            repeat[1].n = real_array_size[1]; repeat[1].is = real_array_size[2]; repeat[1].os = array_size[2];
            plan_x_r2c = FPREF(plan_guru_dft_r2c)(transform_rank, &transform, repeat_rank, repeat, scratchA.real_ptr(), scratchB.fftw_cmpl_ptr(), FFTW_DESTROY_INPUT | FFTW_MEASURE);

            transform.n = array_size[2]; // transform length
            transform.is = transform.os = 1;
            repeat[0].n = array_size[0]; repeat[0].is = array_size[1] * array_size[2]; repeat[0].os = real_array_size[1] * real_array_size[2];
            repeat[1].n = array_size[1]; repeat[1].is = array_size[2]; repeat[1].os = real_array_size[2];
            plan_x_c2r = FPREF(plan_guru_dft_c2r)(transform_rank, &transform, repeat_rank, repeat, scratchA.fftw_cmpl_ptr(), scratchB.real_ptr(), FFTW_DESTROY_INPUT | FFTW_MEASURE);

            // plan for y-directional decomposition
            // ysize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            array_size[0] = scratchA.cmpldec.ysize.z;
            array_size[1] = scratchA.cmpldec.ysize.y;
            array_size[2] = scratchA.cmpldec.ysize.x;
            transform.n = array_size[1]; // transform length
            transform.is = transform.os = array_size[2];
            repeat[0].n = array_size[0]; repeat[0].is = repeat[0].os = array_size[2] * array_size[1];
            repeat[1].n = array_size[2]; repeat[1].is = repeat[1].os = 1;
            plan_y_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.fftw_cmpl_ptr(), scratchB.fftw_cmpl_ptr(), FFTW_FORWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            plan_y_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.fftw_cmpl_ptr(), scratchB.fftw_cmpl_ptr(), FFTW_BACKWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            // plan for z-directional decomposition
            // zsize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            array_size[0] = scratchA.cmpldec.zsize.z;
            array_size[1] = scratchA.cmpldec.zsize.y;
            array_size[2] = scratchA.cmpldec.zsize.x;
            transform.n = array_size[0]; // transform length
            transform.is = transform.os = array_size[1] * array_size[2];
            repeat[0].n = array_size[1]; repeat[0].is = repeat[0].os = array_size[2];
            repeat[1].n = array_size[2]; repeat[1].is = repeat[1].os = 1;
            plan_z_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.fftw_cmpl_ptr(), scratchB.fftw_cmpl_ptr(), FFTW_FORWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            plan_z_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.fftw_cmpl_ptr(), scratchB.fftw_cmpl_ptr(), FFTW_BACKWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
        }
        ~FFTW(){
            FPREF(destroy_plan)(plan_x_r2c);
            FPREF(destroy_plan)(plan_x_c2r);
            FPREF(destroy_plan)(plan_y_forw);
            FPREF(destroy_plan)(plan_y_back);
            FPREF(destroy_plan)(plan_z_forw);
            FPREF(destroy_plan)(plan_z_back);
        }
        void forward(DecompArray<F>& in, DecompArray<F>& out){
            if(in.realdec != out.realdec or in.cmpldec != out.cmpldec)
                ERROR("arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_real() and out.is_cmpl()){
                FPREF(execute_dft_r2c(plan_x_r2c, in.real_ptr(), out.fftw_cmpl_ptr()));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_y_forw, in.fftw_cmpl_ptr(), out.fftw_cmpl_ptr()));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_z_forw, in.fftw_cmpl_ptr(), out.fftw_cmpl_ptr()));
            } else ERROR("array decomposition mismatch in FFT::forward");
        }
        void backward(DecompArray<F>& in, DecompArray<F>& out){
            if(in.realdec != out.realdec or in.cmpldec != out.cmpldec)
                ERROR("arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_cmpl() and out.is_real()){
                FPREF(execute_dft_c2r(plan_x_c2r, in.fftw_cmpl_ptr(), out.real_ptr()));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_y_back, in.fftw_cmpl_ptr(), out.fftw_cmpl_ptr()));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_z_back, in.fftw_cmpl_ptr(), out.fftw_cmpl_ptr()));
            } else ERROR("array decomposition mismatch in FFT::backward");
        }
    };
}

namespace DecompWithFFTW {
    using DecompImpl::Bookkeeping;
    using DecompWithFFTWImpl::DecompArray;
    template <typename F>
    using DistributedFFT = DecompWithFFTWImpl::FFTW<F>;
}
