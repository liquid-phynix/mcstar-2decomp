#pragma once
#include <complex>
#include <vector>
#include <clFFT.h>
#include "decomp.hpp"
#include "sys/mman.h"
//#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

void oclAssert(const char* pref, cl_int err, const char* file, int line){
    if(err == CL_SUCCESS) return;
    const char* reasons[] = {"CL_INVALID_MEM_OBJECT", "CL_INVALID_VALUE", "CL_MISALIGNED_SUB_BUFFER_OFFSET", "CL_MEM_COPY_OVERLAP", "CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY"};
    const int reason_codes[] = {CL_INVALID_MEM_OBJECT, CL_INVALID_VALUE, CL_MISALIGNED_SUB_BUFFER_OFFSET, CL_MEM_COPY_OVERLAP, CL_MEM_OBJECT_ALLOCATION_FAILURE, CL_OUT_OF_RESOURCES, CL_OUT_OF_HOST_MEMORY};
    for(int i = 0; i < sizeof(reasons) / sizeof(reasons[0]); i++){
        if(err == reason_codes[i])
            fprintf(stderr, "OCLERR(%s): %s %s:%d\n", reasons[i], pref, file, line);
        else
            fprintf(stderr, "OCLERR(unknown): %s %s:%d\n", pref, file, line);
        break;
    }
    exit(1);
}
#define OCLERR(arg) oclAssert(#arg, arg, __FILE__, __LINE__);

namespace DecompGlobals {
    cl::Context context;
    cl::CommandQueue queue;
    clfftSetupData* fftSetup = NULL;
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
      //p.getInfo(CL_PLATFORM_PROFILE, &str);
      //std::cout << "profile: " << str << std::endl;
      //p.getInfo(CL_PLATFORM_VERSION, &str);
      //std::cout << "version: " << str << std::endl;
      p.getInfo(CL_PLATFORM_NAME, &str);
      std::cout << "name: " << str << std::endl;
      //p.getInfo(CL_PLATFORM_VENDOR, &str);
      //std::cout << "vendor: " << str << std::endl;
      //p.getInfo(CL_PLATFORM_EXTENSIONS, &str);
      //std::cout << "exts: " << str << std::endl;
      std::cout << "**************\n";
    }

    void print_device_info(cl::Device& d){
      std::cout << "****device****\n";
      std::string str;
      d.getInfo(CL_DEVICE_NAME, &str);
      std::cout << "name: " << str << std::endl;
      //d.getInfo(CL_DEVICE_VENDOR, &str);
      //std::cout << "vendor: " << str << std::endl;
      //d.getInfo(CL_DEVICE_PROFILE, &str);
      //std::cout << "profile: " << str << std::endl;
      //d.getInfo(CL_DEVICE_VERSION, &str);
      //std::cout << "device ver.: " << str << std::endl;
      //d.getInfo(CL_DRIVER_VERSION, &str);
      //std::cout << "driver ver.: " << str << std::endl;
      //d.getInfo(CL_DEVICE_OPENCL_C_VERSION, &str);
      //std::cout << "ocl ver.: " << str << std::endl;
      //d.getInfo(CL_DEVICE_EXTENSIONS, &str);
      //std::cout << "exts: " << str << std::endl;
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
        //ASSERTMSG(devicenum < devices.size(), "devicenum out of range");
        if(devicenum >= devices.size()){devicenum = 0;}
        std::string device_name;
        devices[devicenum].getInfo(CL_DEVICE_NAME, &device_name);
        strncpy(charbuf, device_name.c_str(), sizeof(charbuf));
        ASSERTMSG(check_if_homogeneous(charbuf, sizeof(charbuf)), "device is non-homogeneous among processes, exiting");

        if(rank() == 0){
            print_platform_info(platforms[pf]);
            print_device_info(devices[devicenum]);
        }

        DecompGlobals::queue = cl::CommandQueue(DecompGlobals::context, devices[devicenum]);

        DecompGlobals::fftSetup = new clfftSetupData;
        OCLERR(clfftInitSetupData(DecompGlobals::fftSetup));
        OCLERR(clfftSetup(DecompGlobals::fftSetup));
    }
    void end_decomp_context(){
        if(not DecompGlobals::context_started){
            std::cerr << "decomp context has not been started" << std::endl;
            return;
        }
        //OCLERR(clfftTeardown());
        delete DecompGlobals::fftSetup;
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
                DecompGlobals::queue.finish();
                ASSERTMSG(host_ptr != NULL, "nullptr is returned when mapping buffer");
            }
            ~ContextMan(){
                DecompGlobals::queue.enqueueUnmapMemObject(buffer, host_ptr);
                DecompGlobals::queue.finish();
            }
            template <typename T> operator T*(){ return (T*)host_ptr; }
        };
    public:
        size_t alloc_bytes, y_cont_alloc_bytes;
        cl::Buffer buffer;
        cl::Buffer buffer_y_cont;
        MemoryMan(DecompImpl::DecompInfo di){
            size_t alloc_len = std::max(std::max(di.cmpldec.xsize.prod(), di.cmpldec.ysize.prod()), di.cmpldec.zsize.prod());
            alloc_bytes = sizeof(CT) * alloc_len;
            buffer = cl::Buffer(DecompGlobals::context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, alloc_bytes);
            DecompGlobals::queue.enqueueFillBuffer<unsigned char>(buffer, 0, 0, alloc_bytes, NULL);
            y_cont_alloc_bytes = sizeof(CT) * di.cmpldec.ysize.x * di.cmpldec.ysize.y;
            buffer_y_cont = cl::Buffer(DecompGlobals::context, CL_MEM_READ_WRITE, y_cont_alloc_bytes);
            DecompGlobals::queue.enqueueFillBuffer<unsigned char>(buffer_y_cont, 0, 0, y_cont_alloc_bytes, NULL);
            DecompGlobals::queue.finish();
        }
        cl_mem* get_mem(){ return &buffer.object_; }
        cl_mem* get_y_cont_mem(){ return &buffer_y_cont.object_; }
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
            //int3(di.realdec.xsize.x, di.realdec.xsize.y, di.realdec.xsize.z).size_t3(shape);
            int3(di.realdec.xsize.x, 1, 1).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_x_r2c, DecompGlobals::context.object_, CLFFT_1D, shape));
            OCLERR(clfftSetPlanScale(plan_x_r2c, CLFFT_FORWARD, cl_float(1)));
            OCLERR(clfftSetPlanScale(plan_x_r2c, CLFFT_BACKWARD, cl_float(1)));
            OCLERR(clfftSetPlanBatchSize(plan_x_r2c, di.realdec.xsize.y * di.realdec.xsize.z));
            OCLERR(clfftSetPlanDistance(plan_x_r2c, di.realdec.xsize.x, di.cmpldec.xsize.x));
            // STRIDES
            int3(1, 1, 1).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_x_r2c, CLFFT_1D, shape));
            int3(1, 1, 1).size_t3(shape);
            OCLERR(clfftSetPlanOutStride(plan_x_r2c, CLFFT_1D, shape));

            // plan_x_c2r: INPUT CMPL X-stencil, OUTPUT REAL X-stencil
            //int3(di.realdec.xsize.x, di.realdec.xsize.y, di.realdec.xsize.z).size_t3(shape);
            int3(di.realdec.xsize.x, 1, 1).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_x_c2r, DecompGlobals::context.object_, CLFFT_1D, shape));
            OCLERR(clfftSetPlanScale(plan_x_c2r, CLFFT_FORWARD, cl_float(1)));
            OCLERR(clfftSetPlanScale(plan_x_c2r, CLFFT_BACKWARD, cl_float(1)));
            OCLERR(clfftSetPlanBatchSize(plan_x_c2r, di.realdec.xsize.y * di.realdec.xsize.z));
            OCLERR(clfftSetPlanDistance(plan_x_c2r, di.cmpldec.xsize.x, di.realdec.xsize.x));
            // STRIDES
            int3(1, 1, 1).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_x_c2r, CLFFT_1D, shape));
            int3(1, 1, 1).size_t3(shape);
            OCLERR(clfftSetPlanOutStride(plan_x_c2r, CLFFT_1D, shape));

            // plan_y: INPUT CMPL Y-stencil, OUTPUT CMPL Y-stencil - FFT along dim Y
            //int3(di.cmpldec.ysize.y, di.cmpldec.ysize.x, di.cmpldec.ysize.z).size_t3(shape);
            int3(di.cmpldec.ysize.y, 1, 1).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_y,     DecompGlobals::context.object_, CLFFT_1D, shape));
            OCLERR(clfftSetPlanScale(plan_y, CLFFT_FORWARD, cl_float(1)));
            OCLERR(clfftSetPlanScale(plan_y, CLFFT_BACKWARD, cl_float(1)));
            OCLERR(clfftSetPlanBatchSize(plan_y, di.cmpldec.ysize.x));
            OCLERR(clfftSetPlanDistance(plan_y, 1, 1));
            // STRIDES
            int3(di.cmpldec.ysize.x, 1, 1).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_y,  CLFFT_1D, shape));
            OCLERR(clfftSetPlanOutStride(plan_y, CLFFT_1D, shape));

            // plan_z: INPUT CMPL Z-stencil, OUTPUT CMPL Z-stencil
            //int3(di.cmpldec.zsize.z, di.cmpldec.zsize.x, di.cmpldec.zsize.y).size_t3(shape);
            int3(di.cmpldec.zsize.z, 1, 1).size_t3(shape);
            OCLERR(clfftCreateDefaultPlan(&plan_z, DecompGlobals::context.object_, CLFFT_1D, shape));
            OCLERR(clfftSetPlanScale(plan_z, CLFFT_FORWARD, cl_float(1)));
            OCLERR(clfftSetPlanScale(plan_z, CLFFT_BACKWARD, cl_float(1)));
            OCLERR(clfftSetPlanBatchSize(plan_z, di.cmpldec.zsize.x * di.cmpldec.zsize.y));
            OCLERR(clfftSetPlanDistance(plan_z, 1, 1));
            // STRIDES
            int3(di.cmpldec.zsize.x * di.cmpldec.zsize.y, 1, 1).size_t3(shape);
            OCLERR(clfftSetPlanInStride(plan_z,  CLFFT_1D, shape));
            OCLERR(clfftSetPlanOutStride(plan_z, CLFFT_1D, shape));

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

            DecompGlobals::queue.finish();
        }
        ~DistributedFFT(){
            OCLERR(clfftDestroyPlan(&plan_x_r2c));
            OCLERR(clfftDestroyPlan(&plan_x_c2r));
            OCLERR(clfftDestroyPlan(&plan_y));
            OCLERR(clfftDestroyPlan(&plan_z));
        }
        void forward(DecompArray& in, DecompArray& out){
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");

                   if(in.is_x() and out.is_x() and in.is_real() and out.is_cmpl()){
                OCLERR(clfftEnqueueTransform(plan_x_r2c, CLFFT_FORWARD, 1,
                       &DecompGlobals::queue.object_, 0, NULL, NULL, in.mm.get_mem(), out.mm.get_mem(), NULL));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                size_t stride = in.mm.y_cont_alloc_bytes;
                for(int zi = 0; zi < in.decinfo.cmpldec.ysize.z; zi++){
                    OCLERR(clEnqueueCopyBuffer(DecompGlobals::queue.object_, *in.mm.get_mem(), *in.mm.get_y_cont_mem(), zi * stride, 0, stride, 0, NULL, NULL));
                    OCLERR(clfftEnqueueTransform(plan_y, CLFFT_FORWARD, 1,
                           &DecompGlobals::queue.object_, 0, NULL, NULL, in.mm.get_y_cont_mem(), out.mm.get_y_cont_mem(), NULL));
                    OCLERR(clEnqueueCopyBuffer(DecompGlobals::queue.object_, *out.mm.get_y_cont_mem(), *out.mm.get_mem(), 0, zi * stride, stride, 0, NULL, NULL));
                }
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                OCLERR(clfftEnqueueTransform(plan_z, CLFFT_FORWARD, 1,
                       &DecompGlobals::queue.object_, 0, NULL, NULL, in.mm.get_mem(), out.mm.get_mem(), NULL));
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::forward");

            DecompGlobals::queue.finish();
        }
        void backward(DecompArray& in, DecompArray& out){
            ASSERTMSG(in.decinfo == out.decinfo, "arrays of different decomposition index cannot be transformed");

                   if(in.is_x() and out.is_x() and in.is_cmpl() and out.is_real()){
                OCLERR(clfftEnqueueTransform(plan_x_c2r, CLFFT_BACKWARD, 1,
                       &DecompGlobals::queue.object_, 0, NULL, NULL, in.mm.get_mem(), out.mm.get_mem(), NULL));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                size_t stride = in.mm.y_cont_alloc_bytes;
                for(int zi = 0; zi < in.decinfo.cmpldec.ysize.z; zi++){
                    OCLERR(clEnqueueCopyBuffer(DecompGlobals::queue.object_, *in.mm.get_mem(), *in.mm.get_y_cont_mem(), zi * stride, 0, stride, 0, NULL, NULL));
                    OCLERR(clfftEnqueueTransform(plan_y, CLFFT_BACKWARD, 1,
                           &DecompGlobals::queue.object_, 0, NULL, NULL, in.mm.get_y_cont_mem(), out.mm.get_y_cont_mem(), NULL));
                    OCLERR(clEnqueueCopyBuffer(DecompGlobals::queue.object_, *out.mm.get_y_cont_mem(), *out.mm.get_mem(), 0, zi * stride, stride, 0, NULL, NULL));
                }
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                OCLERR(clfftEnqueueTransform(plan_z, CLFFT_BACKWARD, 1,
                       &DecompGlobals::queue.object_, 0, NULL, NULL, in.mm.get_mem(), out.mm.get_mem(), NULL));
            } else ASSERTMSG(false, "array decomposition mismatch in FFT::backward");

            DecompGlobals::queue.finish();
        }
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
