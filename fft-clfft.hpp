namespace CLFFT {
#define __CL_ENABLE_EXCEPTIONS
#include "utils.hpp"
#include "decomp.hpp"
#include "cl.hpp"
#include <clFFT.h>

inline void oclAssert(const char* pref, cl_int err, const char* file, int line){
    if(err != CL_SUCCESS){
        fprintf(stderr, "OCLERR: %s %s:%d\n", pref, file, line); exit(1);
    }
}
#define OCLERR(arg) oclAssert(#arg, arg, __FILE__, __LINE__);

    template <typename F> class FFT {
    private:
        bool inited;
        cl::Context& context;
        cl::CommandQueue& queue;
        DecompInfo real, cmpl;
        clfftPlanHandle plan_x_r2c, plan_x_c2r, plan_y, plan_z;
    public:
        typedef F                RT;
        typedef std::complex<RT> CT;

        FFT(DecompInfo _real, DecompInfo _cmpl) : real(_real), cmpl(_cmpl){
            inited = false;
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if(pf >= platforms.size()){
            pf = 0;
            std::cerr << "> rank " << rank << " defaulted to platform 0" << std::endl;
        }
        std::string platform_name;
        platforms[pf].getInfo(CL_PLATFORM_NAME, &platform_name);
        strncpy(pname, platform_name.c_str(), sizeof(pname));
        MPI::COMM_WORLD.Gather(pname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, othernames, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, 0);
        if(rank == 0){

            for(int id = 0; id < size; id++){
                if(strncmp(pname, &othernames[id * MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME) != 0){
                    ERROR("platform is non-homogeneous among processes, exiting"); }}}

        cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pf])(), 0 };
        context = cl::Context(devt, cps);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if(devicenum >= devices.size()){
            //ERROR("devicenum out of range");
            devicenum = 0;
        }
        std::string device_name;
        devices[devicenum].getInfo(CL_DEVICE_NAME, &device_name);
        strncpy(pname, device_name.c_str(), sizeof(pname));
        MPI::COMM_WORLD.Gather(pname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, othernames, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, 0);
        if(rank == 0){
            for(int id = 0; id < size; id++){
                if(strncmp(pname, &othernames[id * MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME) != 0){
                    ERROR("device is non-homogeneous among processes, exiting"); }}
            std::cerr << "> selected device <" << device_name << "> from platform <" << platform_name << ">" << std::endl;
            delete[] othernames;
        }
        queue = cl::CommandQueue(context, devices[devicenum]);

        clfftSetupData fftSetup;
        OCLERR(clfftInitSetupData(&fftSetup));
        OCLERR(clfftSetup(&fftSetup));

        }
        void init(RT* real_ptr, CT* _cmpl_ptr, CT* _cmpl_ptr_im){

            inited = true;
        }
        ~FFT(){
            clfftDestroyPlan(&plan_x_r2c);
            clfftDestroyPlan(&plan_x_c2r);
            clfftDestroyPlan(&plan_y);
            clfftDestroyPlan(&plan_z);
            OCLERR(clfftTeardown());
        }
        void execute_x_r2c(){ if(not inited) ERROR("call .init()!"); FPREF(execute(plan_x_r2c)); }
        void execute_x_c2r(){ if(not inited) ERROR("call .init()!"); FPREF(execute(plan_x_c2r)); }
        void execute_y_f(){   if(not inited) ERROR("call .init()!"); FPREF(execute(plan_y_forw)); }
        void execute_y_b(){   if(not inited) ERROR("call .init()!"); FPREF(execute(plan_y_back)); }
        void execute_z_f(){   if(not inited) ERROR("call .init()!"); FPREF(execute(plan_z_forw)); }
        void execute_z_b(){   if(not inited) ERROR("call .init()!"); FPREF(execute(plan_z_back)); }
    };
}
