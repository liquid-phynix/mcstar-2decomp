#pragma once
#include <cstdio>
#include <complex>
#include "decomp.hpp"
#include "sys/mman.h"

namespace CU {

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>

    inline void gpuAssert(cudaError_t code, const char* file, int line){
        if(code != cudaSuccess){
            fprintf(stderr, "CUERR: %s %s:%d\n", cudaGetErrorString(code), file, line);
            fflush(stderr);
            exit(code);
        }
    }
    static const char* cufftGetErrorString(cufftResult error){
        switch(error){
            case CUFFT_SUCCESS:                   return "CUFFT_SUCCESS";
            case CUFFT_INVALID_PLAN:              return "CUFFT_INVALID_PLAN";
            case CUFFT_ALLOC_FAILED:              return "CUFFT_ALLOC_FAILED";
            case CUFFT_INVALID_TYPE:              return "CUFFT_INVALID_TYPE";
            case CUFFT_INVALID_VALUE:             return "CUFFT_INVALID_VALUE";
            case CUFFT_INTERNAL_ERROR:            return "CUFFT_INTERNAL_ERROR";
            case CUFFT_EXEC_FAILED:               return "CUFFT_EXEC_FAILED";
            case CUFFT_SETUP_FAILED:              return "CUFFT_SETUP_FAILED";
            case CUFFT_INVALID_SIZE:              return "CUFFT_INVALID_SIZE";
            case CUFFT_UNALIGNED_DATA:            return "CUFFT_UNALIGNED_DATA";
#if CUDA_VERSION >= 5000
                                                  // newer versions of cufft.h define these too
            case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
            case CUFFT_INVALID_DEVICE:            return "CUFFT_INVALID_DEVICE";
            case CUFFT_PARSE_ERROR:               return "CUFFT_PARSE_ERROR";
            case CUFFT_NO_WORKSPACE:              return "CUFFT_NO_WORKSPACE";
#endif
            default:                              return "<unknown>";
        }
    }
    inline void gpuCufftAssert(cufftResult code, const char* file, int line){
        if(code != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFTERR: %s %s:%d\n", cufftGetErrorString(code), file, line);
            fflush(stderr);
            exit(code);
        }
    }
}
#define CUERR(ans) CU::gpuAssert((ans), __FILE__, __LINE__);
#define CUFFTERR(ans) CU::gpuCufftAssert((ans), __FILE__, __LINE__);

namespace DecompCUFFTImpl {
    using Decomp::RT;
    using Decomp::CT;
    using Decomp::SyncType;
    using DecompImpl::int3;
    using DecompImpl::rank;
    using DecompImpl::rank_in_node;
    using DecompImpl::check_if_homogeneous;

    void start_decomp_context(int3 gshape){
        if(DecompGlobals::context_started){
            std::cerr << "decomp context already started" << std::endl;
            return;
        }
        DecompImpl::start_decomp_context(gshape);

        int devices_on_node;
        CUERR(CU::cudaGetDeviceCount(&devices_on_node));
        if(rank_in_node()+1 > devices_on_node){
            std::cerr << "process " << rank() << " will oversubscribe" << std::endl;
        }
        int device_id = rank_in_node()%devices_on_node;
        CUERR(CU::cudaSetDevice(device_id));

        CU::cudaDeviceProp props;
        CUERR(CU::cudaGetDeviceProperties(&props, device_id));

        ASSERTMSG(check_if_homogeneous(props.name, sizeof(props.name)), "device is non-homogeneous among processes, exiting");

        if(rank() == 0){
            std::cerr << "device is <" << props.name << ">" << std::endl;
        }
    }
    void end_decomp_context(){
        if(not DecompGlobals::context_started){
            std::cerr << "decomp context has not been started" << std::endl;
            return;
        }
        DecompImpl::end_decomp_context();
    }

    class MemoryMan {
    public:
        class ContextMan {
            SyncType st;
            void* host_ptr;
            void* dev_ptr;
            size_t bytes;
        public:
            ContextMan() = delete;
            ContextMan(const ContextMan&) = delete;
            ContextMan(ContextMan&& cm) : host_ptr(std::move(cm.host_ptr)), dev_ptr(std::move(cm.dev_ptr)){};
            ContextMan(SyncType _st, void* _host_ptr, void* _dev_ptr, size_t _bytes) : st(_st), host_ptr(_host_ptr), dev_ptr(_dev_ptr), bytes(_bytes){
                if(st == SyncType::D2H or st == SyncType::BOTH){
                    CUERR(CU::cudaMemcpy(dev_ptr, host_ptr, bytes, CU::cudaMemcpyHostToDevice));
                    CUERR(CU::cudaPeekAtLastError());
                }
            }
            ~ContextMan(){
                if(st == SyncType::H2D or st == SyncType::BOTH){
                    CUERR(CU::cudaMemcpy(host_ptr, dev_ptr, bytes, CU::cudaMemcpyDeviceToHost));
                    CUERR(CU::cudaPeekAtLastError());
                }
            }
            template <typename T> operator T*(){ return (T*)host_ptr; }
        };
    private:
        size_t alloc_bytes;
        CT* host_ptr;
        CT* dev_ptr;
    public:
        MemoryMan(DecompImpl::DecompInfo di){
            size_t alloc_len = std::max(std::max(di.cmpldec.xsize.prod(), di.cmpldec.ysize.prod()), di.cmpldec.zsize.prod());
            alloc_bytes = sizeof(CT) * alloc_len;
            CUERR(CU::cudaMalloc((void**)&dev_ptr, alloc_bytes));
            CUERR(CU::cudaHostAlloc((void**)&host_ptr, alloc_bytes, cudaHostAllocDefault));
            CUERR(CU::cudaPeekAtLastError());
            if(mlock(host_ptr, alloc_bytes) != 0)
                std::cerr << "MemoryMan: host memory not pinned" << std::endl;
            memset(host_ptr, 0, alloc_bytes);
            CUERR(CU::cudaMemset(dev_ptr, 0, alloc_bytes));
            CUERR(CU::cudaPeekAtLastError());
        }
        ~MemoryMan(){
            CUERR(CU::cudaFreeHost(host_ptr));
            CUERR(CU::cudaFree(dev_ptr));
            CUERR(CU::cudaPeekAtLastError());
        }
        template <typename T> operator T*(){ return (T*)host_ptr; }
        ContextMan operator()(SyncType st) const { return ContextMan(st, host_ptr, dev_ptr, alloc_bytes); }
    };

    using DecompArray = DecompImpl::DecompArray<MemoryMan>;

    class DistributedFFT : public DecompImpl::DistributedFFTBase<DistributedFFT> {
        CU::cufftHandle plan_x_r2c, plan_x_c2r, plan_y, plan_z;
        int y_stride, y_rep;
    public:
        DistributedFFT() = delete;
        DistributedFFT(const DistributedFFT&) = delete;
        DistributedFFT(DecompImpl::DecompInfo di){
            int n[1];
            // plan for x-directional decomposition
            n[0] = di.realdec.xsize.x;
            CUFFTERR(CU::cufftPlanMany(&plan_x_r2c, 1, n,
                                NULL, 1, n[0], NULL, 1, n[0],
                                sizeof(RT)==sizeof(float) ? CU::CUFFT_R2C : CU::CUFFT_D2Z, di.realdec.xsize.y * di.realdec.xsize.z));
            CUFFTERR(CU::cufftPlanMany(&plan_x_c2r, 1, n,
                                NULL, 1, n[0], NULL, 1, n[0],
                                sizeof(RT)==sizeof(float) ? CU::CUFFT_C2R : CU::CUFFT_Z2D, di.realdec.xsize.y * di.realdec.xsize.z));

            // plan for y-directional decomposition
            n[0] = di.cmpldec.ysize.y;
            CUFFTERR(CU::cufftPlanMany(&plan_y, 1, n,
                                NULL, di.cmpldec.ysize.x, 1,
                                NULL, di.cmpldec.ysize.x, 1,
                                sizeof(RT)==sizeof(float) ? CU::CUFFT_C2C : CU::CUFFT_Z2Z, di.cmpldec.ysize.x));
            y_stride = di.cmpldec.ysize.x * di.cmpldec.ysize.y;
            y_rep = di.cmpldec.ysize.z;

            // plan for z-directional decomposition
            n[0] = di.cmpldec.zsize.z;
            CUFFTERR(CU::cufftPlanMany(&plan_z, 1, n,
                                NULL, di.cmpldec.zsize.x * di.cmpldec.zsize.y, 1,
                                NULL, di.cmpldec.zsize.x * di.cmpldec.zsize.y, 1,
                                sizeof(RT)==sizeof(float) ? CU::CUFFT_C2C : CU::CUFFT_Z2Z, di.cmpldec.zsize.x * di.cmpldec.zsize.y));
        }
        ~DistributedFFT(){
            CUFFTERR(CU::cufftDestroy(plan_x_r2c));
            CUFFTERR(CU::cufftDestroy(plan_x_c2r));
            CUFFTERR(CU::cufftDestroy(plan_y));
            CUFFTERR(CU::cufftDestroy(plan_z));
        }
        void forward(DecompArray& in, DecompArray& out){
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_real() and out.is_cmpl()){
                if(sizeof(RT) == sizeof(float)){
                    CUFFTERR(CU::cufftExecR2C(plan_x_r2c, in.mm, out.mm));
                } else if(sizeof(RT) == sizeof(double)){
                    CUFFTERR(CU::cufftExecD2Z(plan_x_r2c, in.mm, out.mm));
                } else ASSERTMSG(false, "cannot happen");
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                if(sizeof(RT) == sizeof(float)){
                    for(int i = 0; i < y_rep; i++)
                        CUFFTERR(CU::cufftExecC2C(plan_y, ((CU::float2*)in.mm)+i*y_stride, ((CU::float2*)out.mm)+i*y_stride, CUFFT_FORWARD));
                } else if(sizeof(RT) == sizeof(double)){
                    for(int i = 0; i < y_rep; i++)
                        CUFFTERR(CU::cufftExecZ2Z(plan_y, ((CU::double2*)in.mm)+i*y_stride, ((CU::double2*)out.mm)+i*y_stride, CUFFT_FORWARD));
                } else ASSERTMSG(false, "cannot happen");
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                if(sizeof(RT) == sizeof(float)){
                    CUFFTERR(CU::cufftExecC2C(plan_z, in.mm, out.mm, CUFFT_FORWARD));
                } else if(sizeof(RT) == sizeof(double)){
                    CUFFTERR(CU::cufftExecZ2Z(plan_z, in.mm, out.mm, CUFFT_FORWARD));
                } else ASSERTMSG(false, "cannot happen");
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::forward");
            CUERR(CU::cudaThreadSynchronize());
        }
        void backward(DecompArray& in, DecompArray& out){
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_cmpl() and out.is_real()){
                if(sizeof(RT) == sizeof(float)){
                    CUFFTERR(CU::cufftExecC2R(plan_x_c2r, in.mm, out.mm));
                } else if(sizeof(RT) == sizeof(double)){
                    CUFFTERR(CU::cufftExecZ2D(plan_x_c2r, in.mm, out.mm));
                } else ASSERTMSG(false, "cannot happen");
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                if(sizeof(RT) == sizeof(float)){
                    for(int i = 0; i < y_rep; i++)
                        CUFFTERR(CU::cufftExecC2C(plan_y, ((CU::float2*)in.mm)+i*y_stride, ((CU::float2*)out.mm)+i*y_stride, CUFFT_INVERSE));
                } else if(sizeof(RT) == sizeof(double)){
                    for(int i = 0; i < y_rep; i++)
                        CUFFTERR(CU::cufftExecZ2Z(plan_y, ((CU::double2*)in.mm)+i*y_stride, ((CU::double2*)out.mm)+i*y_stride, CUFFT_INVERSE));
                } else ASSERTMSG(false, "cannot happen");
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                if(sizeof(RT) == sizeof(float)){
                    CUFFTERR(CU::cufftExecC2C(plan_z, in.mm, out.mm, CUFFT_INVERSE));
                } else if(sizeof(RT) == sizeof(double)){
                    CUFFTERR(CU::cufftExecZ2Z(plan_z, in.mm, out.mm, CUFFT_INVERSE));
                } else ASSERTMSG(false, "cannot happen");
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::backward");
            CUERR(CU::cudaThreadSynchronize());
        }
    };
}

namespace DecompCUFFT {
    using Decomp::RT;
    using Decomp::CT;
    using DecompImpl::int3;
    using DecompImpl::over;
    //using DecompImpl::size;
    //using DecompImpl::rank;
    //using DecompImpl::seed;
    //using DecompImpl::rank_in_node;
    //using DecompImpl::size_of_node;
    using DecompCUFFTImpl::start_decomp_context;
    using DecompCUFFTImpl::end_decomp_context;
    using DecompCUFFTImpl::DecompArray;
    using DecompCUFFTImpl::DistributedFFT;
}
