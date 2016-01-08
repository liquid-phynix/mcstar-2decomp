#pragma once
#include <complex>
#include <vector>
#include <clFFT.h>
#include "decomp.hpp"
#include "sys/mman.h"
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

void oclAssert(const char* pref, cl_int err, const char* file, int line){
    if(err != CL_SUCCESS){
        fprintf(stderr, "OCLERR: %s %s:%d\n", pref, file, line); exit(1);
    }
}
#define OCLERR(arg) oclAssert(#arg, arg, __FILE__, __LINE__);

namespace DecompGlobals {
    cl::Context context;
    cl::CommandQueue queue;
    clfftSetupData fftSetup;
}

namespace DecompCLFFTImpl {
    using Decomp::RT;
    using Decomp::CT;
    using DecompImpl::int3;
    using DecompImpl::rank;
    using DecompImpl::rank_in_node;
    using DecompImpl::check_if_homogeneous;

    static const cl_device_type ALL = CL_DEVICE_TYPE_ALL;
    static const cl_device_type GPU = CL_DEVICE_TYPE_GPU;
    static const cl_device_type CPU = CL_DEVICE_TYPE_CPU;

    void print_platform_info(cl::Platform& p){
      std::cout << "***platform***\n";
      std::string str;
      p.getInfo(CL_PLATFORM_PROFILE, &str);
      std::cout << "profile: " << str << std::endl;
      p.getInfo(CL_PLATFORM_VERSION, &str);
      std::cout << "version: " << str << std::endl;
      p.getInfo(CL_PLATFORM_NAME, &str);
      std::cout << "name: " << str << std::endl;
      p.getInfo(CL_PLATFORM_VENDOR, &str);
      std::cout << "vendor: " << str << std::endl;
      p.getInfo(CL_PLATFORM_EXTENSIONS, &str);
      std::cout << "exts: " << str << std::endl;
      std::cout << "**************\n";
    }

    void print_device_info(cl::Device& d){
      std::cout << "****device****\n";
      std::string str;
      d.getInfo(CL_DEVICE_NAME, &str);
      std::cout << "name: " << str << std::endl;
      d.getInfo(CL_DEVICE_VENDOR, &str);
      std::cout << "vendor: " << str << std::endl;
      d.getInfo(CL_DEVICE_PROFILE, &str);
      std::cout << "profile: " << str << std::endl;
      d.getInfo(CL_DEVICE_VERSION, &str);
      std::cout << "device ver.: " << str << std::endl;
      d.getInfo(CL_DRIVER_VERSION, &str);
      std::cout << "driver ver.: " << str << std::endl;
      d.getInfo(CL_DEVICE_OPENCL_C_VERSION, &str);
      std::cout << "ocl ver.: " << str << std::endl;
      d.getInfo(CL_DEVICE_EXTENSIONS, &str);
      std::cout << "exts: " << str << std::endl;
      std::cout << "**************\n";
    }

    void start_decomp_context(int3 gshape, int pf = 0, cl_device_type devt = CPU){
        if(DecompGlobals::context_started){
            std::cerr << "decomp context already started" << std::endl;
            return;
        }
        DecompImpl::start_decomp_context(gshape);

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(pf >= platforms.size()){
            pf = 0;
            std::cerr << "> rank " << rank() << " defaulted to platform 0" << std::endl;
        }
        std::string platform_name;
        platforms[pf].getInfo(CL_PLATFORM_NAME, &platform_name);
        char charbuf[128];
        strncpy(charbuf, platform_name.c_str(), sizeof(charbuf));
        ASSERTMSG(check_if_homogeneous(charbuf, sizeof(charbuf)), "platform is non-homogeneous among processes, exiting");

        cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pf])(), 0 };
        DecompGlobals::context = cl::Context(devt, cps);
        std::vector<cl::Device> devices = DecompGlobals::context.getInfo<CL_CONTEXT_DEVICES>();
        int devicenum = rank_in_node();
        if(devicenum >= devices.size()){
            //ERROR("devicenum out of range");
            devicenum = 0;
        }
        std::string device_name;
        devices[devicenum].getInfo(CL_DEVICE_NAME, &device_name);
        strncpy(charbuf, device_name.c_str(), sizeof(charbuf));
        ASSERTMSG(check_if_homogeneous(charbuf, sizeof(charbuf)), "device is non-homogeneous among processes, exiting");

        if(rank() == 0){
            print_platform_info(platforms[pf]);
            print_device_info(devices[devicenum]);
        }

        DecompGlobals::queue = cl::CommandQueue(DecompGlobals::context, devices[devicenum]);

        //clfftSetupData fftSetup;
        OCLERR(clfftInitSetupData(&DecompGlobals::fftSetup));
        OCLERR(clfftSetup(&DecompGlobals::fftSetup));
    }
    void end_decomp_context(){
        if(not DecompGlobals::context_started){
            std::cerr << "decomp context has not been started" << std::endl;
            return;
        }
        //OCLERR(clfftTeardown());
        DecompImpl::end_decomp_context();
    }

    class MemoryMan {
    public:
        class ContextMan {
            cl::Buffer& buffer;
            void* host_ptr;
        public:
            ContextMan() = delete;
            ContextMan(const ContextMan&) = delete;
            ContextMan(ContextMan&& cm) : host_ptr(std::move(cm.host_ptr)), buffer(cm.buffer){};
            ContextMan(cl::Buffer& _buffer, size_t bytes) : buffer(_buffer){
                host_ptr = DecompGlobals::queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes);
                int lock = mlock(host_ptr, bytes);
                if(lock) fprintf(stderr, "memory region (mapbuffer) is not pinned\n");
                DecompGlobals::queue.enqueueBarrierWithWaitList();
                DecompGlobals::queue.flush();
                ASSERTMSG(host_ptr != NULL, "nullptr is returned when mapping buffer");
            }
            ~ContextMan(){
                DecompGlobals::queue.enqueueUnmapMemObject(buffer, host_ptr);
                DecompGlobals::queue.enqueueBarrierWithWaitList();
                DecompGlobals::queue.flush();
            }
            template <typename T> operator T*(){ return (T*)host_ptr; }
        };
    public:
        size_t alloc_bytes;
        cl::Buffer buffer;
        MemoryMan(DecompImpl::DecompInfo di){
            size_t alloc_len = std::max(std::max(di.cmpldec.xsize.prod(), di.cmpldec.ysize.prod()), di.cmpldec.zsize.prod());
            alloc_bytes = sizeof(CT) * alloc_len;
            buffer = cl::Buffer(DecompGlobals::context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, alloc_bytes);
            DecompGlobals::queue.enqueueFillBuffer<unsigned char>(buffer, 0, 0, alloc_bytes, NULL);
            DecompGlobals::queue.enqueueBarrierWithWaitList();
            DecompGlobals::queue.flush();
        }
        ContextMan operator()() { return ContextMan(buffer, alloc_bytes); }
    };

    using DecompArray = DecompImpl::DecompArray<MemoryMan>;

    class DistributedFFT : public DecompImpl::DistributedFFTBase<DistributedFFT> {
        clfftPlanHandle plan_x_r2c, plan_x_c2r, plan_y, plan_z;
    public:
        DistributedFFT() = delete;
        DistributedFFT(const DistributedFFT&) = delete;
        DistributedFFT(DecompImpl::DecompInfo di){
            size_t shape[3];
            // plan_x_r2c: INPUT REAL X-stencil, OUTPUT CMPL X-stencil
            int3(di.realdec.xsize.x, di.realdec.xsize.y, di.realdec.xsize.z).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_x_r2c, DecompGlobals::context.object_, CLFFT_1D, shape));
            // STRIDES
            int3(1, di.realdec.xsize.x, di.realdec.xsize.x * di.realdec.xsize.y).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_x_r2c, CLFFT_3D, shape));
            int3(1, di.cmpldec.xsize.x, di.cmpldec.xsize.x * di.cmpldec.xsize.y).size_t3(shape);
            OCLERR(clfftSetPlanOutStride(plan_x_r2c, CLFFT_3D, shape));

            // plan_x_c2r: INPUT CMPL X-stencil, OUTPUT REAL X-stencil
            int3(di.realdec.xsize.x, di.realdec.xsize.y, di.realdec.xsize.z).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_x_c2r, DecompGlobals::context.object_, CLFFT_1D, shape));
            // STRIDES
            int3(1, di.cmpldec.xsize.x, di.cmpldec.xsize.x * di.cmpldec.xsize.y).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_x_c2r, CLFFT_3D, shape));
            int3(1, di.realdec.xsize.x, di.realdec.xsize.x * di.realdec.xsize.y).size_t3(shape);
            OCLERR(clfftSetPlanOutStride(plan_x_c2r, CLFFT_3D, shape));

            // plan_y: INPUT CMPL Y-stencil, OUTPUT CMPL Y-stencil - FFT along dim Y
            int3(di.cmpldec.ysize.y, di.cmpldec.ysize.x, di.cmpldec.ysize.z).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_y,     DecompGlobals::context.object_, CLFFT_1D, shape));
            // STRIDES
            int3(di.cmpldec.ysize.x, 1, di.cmpldec.ysize.x * di.cmpldec.ysize.y).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_y,  CLFFT_3D, shape));
            OCLERR(clfftSetPlanOutStride(plan_y, CLFFT_3D, shape));

            // plan_z: INPUT CMPL Z-stencil, OUTPUT CMPL Z-stencil
            int3(di.cmpldec.zsize.z, di.cmpldec.zsize.x, di.cmpldec.zsize.y).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_z,     DecompGlobals::context.object_, CLFFT_1D, shape));
            // STRIDES
            int3(di.cmpldec.zsize.x * di.cmpldec.zsize.y, 1, di.cmpldec.zsize.x).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_y,  CLFFT_3D, shape));
            OCLERR(clfftSetPlanOutStride(plan_y, CLFFT_3D, shape));

            clfftPrecision prec = sizeof(RT) == 4 ? CLFFT_SINGLE : CLFFT_DOUBLE;
            //clfftPrecision prec = precbytes == 4 ? CLFFT_SINGLE_FAST : CLFFT_DOUBLE_FAST;

            OCLERR(clfftSetPlanPrecision(plan_x_r2c, prec));
            OCLERR(clfftSetPlanPrecision(plan_x_c2r, prec));
            OCLERR(clfftSetPlanPrecision(plan_y,     prec));
            OCLERR(clfftSetPlanPrecision(plan_z,     prec));

            OCLERR(clfftSetLayout(plan_x_r2c, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED));
            OCLERR(clfftSetLayout(plan_x_c2r, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL));
            OCLERR(clfftSetLayout(plan_y, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));
            OCLERR(clfftSetLayout(plan_z, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED));

            OCLERR(clfftSetResultLocation(plan_x_r2c, CLFFT_OUTOFPLACE));
            OCLERR(clfftSetResultLocation(plan_x_c2r, CLFFT_OUTOFPLACE));
            OCLERR(clfftSetResultLocation(plan_y,     CLFFT_OUTOFPLACE));
            OCLERR(clfftSetResultLocation(plan_z,     CLFFT_OUTOFPLACE));

            OCLERR(clfftBakePlan(plan_x_r2c, 1, &DecompGlobals::queue.object_, NULL, NULL));
            OCLERR(clfftBakePlan(plan_x_c2r, 1, &DecompGlobals::queue.object_, NULL, NULL));
            OCLERR(clfftBakePlan(plan_y,     1, &DecompGlobals::queue.object_, NULL, NULL));
            OCLERR(clfftBakePlan(plan_z,     1, &DecompGlobals::queue.object_, NULL, NULL));

            DecompGlobals::queue.enqueueBarrierWithWaitList();
        }
        ~DistributedFFT(){
            OCLERR(clfftDestroyPlan(&plan_x_r2c));
            OCLERR(clfftDestroyPlan(&plan_x_c2r));
            OCLERR(clfftDestroyPlan(&plan_y));
            OCLERR(clfftDestroyPlan(&plan_z));
        }
        void forward(DecompArray& in, DecompArray& out){
            auto manIn = in.mm();
            auto manOut = out.mm();
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_real() and out.is_cmpl()){
                //FPREF(execute_dft_r2c(plan_x_r2c, manIn, manOut));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                //FPREF(execute_dft(plan_y_forw, manIn, manOut));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                //FPREF(execute_dft(plan_z_forw, manIn, manOut));
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::forward");
        }
        void backward(DecompArray& in, DecompArray& out){
            auto manIn = in.mm();
            auto manOut = out.mm();
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_cmpl() and out.is_real()){
                //FPREF(execute_dft_c2r(plan_x_c2r, manIn, manOut));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                //FPREF(execute_dft(plan_y_back, manIn, manOut));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                //FPREF(execute_dft(plan_z_back, manIn, manOut));
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::backward");
        }
    //void execute_x_r2c(cl_mem* in, cl_mem* out){
        //OCLERR(clfftEnqueueTransform(plan_x_r2c, CLFFT_FORWARD, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        //queue.finish();
    //}
    //void execute_x_c2r(cl_mem* in, cl_mem* out){
        //OCLERR(clfftEnqueueTransform(plan_x_c2r, CLFFT_BACKWARD, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        //queue.finish();
    //}
    //void execute_y(cl_mem* in, cl_mem* out, clfftDirection dir){
        //OCLERR(clfftEnqueueTransform(plan_y, dir, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        //queue.finish();
    //}
    //void execute_z(cl_mem* in, cl_mem* out, clfftDirection dir){
        //OCLERR(clfftEnqueueTransform(plan_z, dir, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        //queue.finish();
    //}
    };
}

namespace DecompCLFFT {
    using Decomp::RT;
    using Decomp::CT;
    using DecompImpl::int3;
    using DecompImpl::over;
    //using DecompImpl::size;
    //using DecompImpl::rank;
    //using DecompImpl::seed;
    //using DecompImpl::rank_in_node;
    //using DecompImpl::size_of_node;
    using DecompCLFFTImpl::ALL;
    using DecompCLFFTImpl::GPU;
    using DecompCLFFTImpl::CPU;
    using DecompCLFFTImpl::start_decomp_context;
    using DecompCLFFTImpl::end_decomp_context;
    using DecompCLFFTImpl::DecompArray;
    using DecompCLFFTImpl::DistributedFFT;
}
