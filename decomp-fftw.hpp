#pragma once
#include <fftw3.h>
#include <complex>
#include "decomp.hpp"
#include "sys/mman.h"

namespace DecompFFTWImpl {
    using Decomp::RT;
    using Decomp::CT;

#ifdef SINGLEFLOAT
#define FPREF(a) fftwf_##a
#else
#define FPREF(a) fftw_##a
#endif

    class MemoryMan {
    public:
        class ContextMan {
            CT* const ptr;
        public:
            ContextMan() = delete;
            ContextMan(const ContextMan&) = delete;
            ContextMan(CT* _ptr) : ptr(_ptr){}
            ContextMan(ContextMan&& cm) : ptr(std::move(cm.ptr)){}
            template <typename T> operator T*(){ return (T*)ptr; }
        };
        CT* ptr;
    public:
        size_t alloc_bytes;
        MemoryMan(DecompImpl::DecompInfo di){
            size_t alloc_len = std::max(std::max(di.cmpldec.xsize.prod(), di.cmpldec.ysize.prod()), di.cmpldec.zsize.prod());
            alloc_bytes = sizeof(CT) * alloc_len;
            ptr = (CT*)FPREF(malloc)(alloc_bytes);
            int lock = mlock(ptr, alloc_bytes);
            if(lock) fprintf(stderr, "memory region cannot be pinned\n");
            memset(ptr, 0, alloc_bytes);
        }
        ~MemoryMan(){ FPREF(free)(ptr); }
        ContextMan operator()() const { return ContextMan(ptr); }
    };

    using DecompArray = DecompImpl::DecompArray<MemoryMan>;

    class DistributedFFT : public DecompImpl::DistributedFFTBase<DistributedFFT> {
        FPREF(plan) plan_x_r2c, plan_x_c2r, plan_y_forw, plan_y_back, plan_z_forw, plan_z_back;
    public:
        DistributedFFT() = delete;
        DistributedFFT(const DistributedFFT&) = delete;
        DistributedFFT(DecompArray& scratchA, DecompArray& scratchB){
            ASSERTMSG(sizeof(FPREF(complex)) == sizeof(CT), "main() and fftw float types differ");
            FPREF(iodim) transform{}, repeat[3]{};
            int transform_rank = 1; // transform dimension
            int repeat_rank = 2;    // 2D repetition
            int real_array_size[3]{}; int array_size[3]{};

            auto manA = scratchA.mm();
            auto manB = scratchB.mm();

            // plan for x-directional decomposition
            // xsize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            real_array_size[0] = scratchA.decinfo.realdec.xsize.z;
            real_array_size[1] = scratchA.decinfo.realdec.xsize.y;
            real_array_size[2] = scratchA.decinfo.realdec.xsize.x;
            array_size[0]      = scratchA.decinfo.cmpldec.xsize.z;
            array_size[1]      = scratchA.decinfo.cmpldec.xsize.y;
            array_size[2]      = scratchA.decinfo.cmpldec.xsize.x;

            transform.n = real_array_size[2]; // transform length
            transform.is = transform.os = 1;
            repeat[0].n = real_array_size[0]; repeat[0].is = real_array_size[1] * real_array_size[2]; repeat[0].os = array_size[1] * array_size[2];
            repeat[1].n = real_array_size[1]; repeat[1].is = real_array_size[2]; repeat[1].os = array_size[2];
            plan_x_r2c = FPREF(plan_guru_dft_r2c)(transform_rank, &transform, repeat_rank, repeat, manA, manB, FFTW_DESTROY_INPUT | FFTW_MEASURE);

            //transform.n = array_size[2]; // transform length
            transform.n = real_array_size[2]; // transform length
            transform.is = transform.os = 1;
            repeat[0].n = array_size[0]; repeat[0].is = array_size[1] * array_size[2]; repeat[0].os = real_array_size[1] * real_array_size[2];
            repeat[1].n = array_size[1]; repeat[1].is = array_size[2]; repeat[1].os = real_array_size[2];
            plan_x_c2r = FPREF(plan_guru_dft_c2r)(transform_rank, &transform, repeat_rank, repeat, manA, manB, FFTW_DESTROY_INPUT | FFTW_MEASURE);

            // plan for y-directional decomposition
            // ysize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            array_size[0] = scratchA.decinfo.cmpldec.ysize.z;
            array_size[1] = scratchA.decinfo.cmpldec.ysize.y;
            array_size[2] = scratchA.decinfo.cmpldec.ysize.x;
            transform.n = array_size[1]; // transform length
            transform.is = transform.os = array_size[2];
            repeat[0].n = array_size[0]; repeat[0].is = repeat[0].os = array_size[2] * array_size[1];
            repeat[1].n = array_size[2]; repeat[1].is = repeat[1].os = 1;
            plan_y_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, manA, manB, FFTW_FORWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            plan_y_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, manA, manB, FFTW_BACKWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            // plan for z-directional decomposition
            // zsize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            array_size[0] = scratchA.decinfo.cmpldec.zsize.z;
            array_size[1] = scratchA.decinfo.cmpldec.zsize.y;
            array_size[2] = scratchA.decinfo.cmpldec.zsize.x;
            transform.n = array_size[0]; // transform length
            transform.is = transform.os = array_size[1] * array_size[2];
            repeat[0].n = array_size[1]; repeat[0].is = repeat[0].os = array_size[2];
            repeat[1].n = array_size[2]; repeat[1].is = repeat[1].os = 1;
            plan_z_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, manA, manB, FFTW_FORWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            plan_z_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, manA, manB, FFTW_BACKWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
        }
        ~DistributedFFT(){
            FPREF(destroy_plan)(plan_x_r2c);
            FPREF(destroy_plan)(plan_x_c2r);
            FPREF(destroy_plan)(plan_y_forw);
            FPREF(destroy_plan)(plan_y_back);
            FPREF(destroy_plan)(plan_z_forw);
            FPREF(destroy_plan)(plan_z_back);
        }
        void forward(DecompArray& in, DecompArray& out){
            auto manIn = in.mm();
            auto manOut = out.mm();
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_real() and out.is_cmpl()){
                FPREF(execute_dft_r2c(plan_x_r2c, manIn, manOut));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_y_forw, manIn, manOut));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_z_forw, manIn, manOut));
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::forward");
        }
        void backward(DecompArray& in, DecompArray& out){
            auto manIn = in.mm();
            auto manOut = out.mm();
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_cmpl() and out.is_real()){
                FPREF(execute_dft_c2r(plan_x_c2r, manIn, manOut));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_y_back, manIn, manOut));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_z_back, manIn, manOut));
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::backward");
        }
    };
}

namespace DecompFFTW {
    using Decomp::RT;
    using Decomp::CT;
    using DecompImpl::int3;
    using DecompImpl::start_decomp_context;
    using DecompImpl::end_decomp_context;
    //using DecompImpl::size;
    //using DecompImpl::rank;
    //using DecompImpl::seed;
    //using DecompImpl::rank_in_node;
    //using DecompImpl::size_of_node;
    using DecompFFTWImpl::DecompArray;
    using DecompImpl::over;
    using DecompImpl::DistributedFFT;
}
