#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <vector>
#include <string>
#include <functional>
#include <mpi.h>
#include <unistd.h>
#include <sys/mman.h>

#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#include <clFFT.h>
#include "interop.h"


#define ERROR(msg) { std::fprintf(stderr, "ERR @ %s:%d > %s\n", __FILE__, __LINE__, msg); std::exit(EXIT_FAILURE); }
inline void oclAssert(const char* pref, cl_int err, const char* file, int line){
    if(err != CL_SUCCESS){
        fprintf(stderr, "OCLERR: %s %s:%d\n", pref, file, line); exit(1);
    }
}
#define OCLERR(arg) oclAssert(#arg, arg, __FILE__, __LINE__);

struct int3 {
    typedef int* IPTR;
    int x,y,z;
    operator IPTR(){ return IPTR(this); }
} __attribute__((packed));
std::ostream& operator<<(std::ostream& os, const int3& i3){
    os << "(" << i3.x << "," << i3.y << "," << i3.z << ")";
    return os;
}

template<typename RT>
class Bookkeeping {
    public:
        //static const cl_device_type ALL = CL_DEVICE_TYPE_ALL, GPU = CL_DEVICE_TYPE_GPU, CPU = CL_DEVICE_TYPE_CPU;
        int size, rank;
        cl::Context context;
        cl::CommandQueue queue;
    Bookkeeping(int pf, cl_device_type devt, int3 gshape){
        MPI::Init();
        size = MPI::COMM_WORLD.Get_size();
        rank = MPI::COMM_WORLD.Get_rank();
        char pname[MPI_MAX_PROCESSOR_NAME];
        int pnamelen;
        MPI::Get_processor_name(pname, pnamelen);
        //std::cerr << "rank " << rank << " is named " << pname << std::endl;

        char* othernames = NULL;
        if(rank == 0) othernames = new char[size * MPI_MAX_PROCESSOR_NAME];
        MPI::COMM_WORLD.Gather(pname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, othernames, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, 0);
        int id_by_rank[size];
        if(rank == 0){
            //for(int i = 0; i < size; i++)
            //std::cerr << "> rank " << i << " is named |" << &othernames[i * MPI_MAX_PROCESSOR_NAME] << "|" << std::endl;
            for(int i = 0; i < size; i++) id_by_rank[i] = -1;
            for(int id = 0; id < size; id++){
                if(id_by_rank[id] != -1) continue;
                int pairid = 0;
                id_by_rank[id] = pairid;
                for(int other = 0; other < size; other++){
                    if(id_by_rank[other] != -1) continue;
                    if(0 == strcmp(&othernames[id * MPI_MAX_PROCESSOR_NAME], &othernames[other * MPI_MAX_PROCESSOR_NAME]))
                        id_by_rank[other] = ++pairid;
                }
            }
            //for(int i = 0; i < size; i++)
            //std::cerr << "> rank " << i << " got id " << id_by_rank[i] << std::endl;
        }
        MPI::COMM_WORLD.Scatter(id_by_rank, 1, MPI_INT, &id_by_rank[rank], 1, MPI_INT, 0);
        int devicenum = id_by_rank[rank];

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

        int proc_grid[2] = {0, 0};
        initialize(gshape, proc_grid);
        int float_size;
        get_float_size(&float_size);
        if(float_size != sizeof(RT)){
            fprintf(stderr, "sizeof(F) = %d , mytype_bytes = %d\n", sizeof(RT), float_size);
            ERROR("main() and 2DECOMP float sizes differ");
        }
    }
    ~Bookkeeping(){
        finalize();
        MPI::Finalize();
        //MPI::Finalize();
        //OCLERR(clfftTeardown());
    }
};

struct DecompInfo {
    int float_size;
    int decomp_info_fortran_index, decomp_info_rank;
    int3 xstart, xend, xsize;
    int3 ystart, yend, ysize;
    int3 zstart, zend, zsize;
    DecompInfo() = delete;
    DecompInfo(const DecompInfo& other){
        float_size = other.float_size;
        decomp_info_fortran_index = other.decomp_info_fortran_index;
        decomp_info_rank = other.decomp_info_rank;
        xstart=other.xstart;xend=other.xend;xsize=other.xsize;
        ystart=other.ystart;yend=other.yend;ysize=other.ysize;
        zstart=other.zstart;zend=other.zend;zsize=other.zsize;
    }
    DecompInfo(int3 shape){
        int3 ret[3 * 3] = {};
        if(shape.x <= 1 || shape.y <= 1 || shape.z <= 1)
            ERROR("dimensions must be > 1");
        create_decomp_info(shape, &decomp_info_fortran_index, &decomp_info_rank, (int*)ret);
        get_float_size(&float_size);
        xstart = ret[0]; xend = ret[1]; xsize = ret[2];
        ystart = ret[3]; yend = ret[4]; ysize = ret[5];
        zstart = ret[6]; zend = ret[7]; zsize = ret[8];
    }
    friend std::ostream& operator<<(std::ostream&, const DecompInfo&);
};
std::ostream& operator<<(std::ostream& os, const DecompInfo& di){
    os << "decomp info #" << di.decomp_info_fortran_index << " @ rank " << di.decomp_info_rank << "\n";
    os << "x-pencil from " << di.xstart << "\tto " << di.xend << ",\tsize " << di.xsize << "\n";
    os << "y-pencil from " << di.ystart << "\tto " << di.yend << ",\tsize " << di.ysize << "\n";
    os << "z-pencil from " << di.zstart << "\tto " << di.zend << ",\tsize " << di.zsize << "\n";
    return os;
}

template <typename F>
class Array {
public:
    enum PencilType {XD = 1, YD, ZD};
    PencilType pt;
    int alloc_bytes;
    F* ptr;
    const DecompInfo di;
public:
    void set_x_pencil(){ pt = XD; }
    void set_y_pencil(){ pt = YD; }
    void set_z_pencil(){ pt = ZD; }
    Array(DecompInfo _di, PencilType _pt = XD): di(_di), ptr(NULL), alloc_bytes(0), pt(_pt){
        // calc. max number of array elements from the 3 decompositions X,Y,Z
        int lx = di.xsize.x * di.xsize.y * di.xsize.z;
        int ly = di.ysize.x * di.ysize.y * di.ysize.z;
        int lz = di.zsize.x * di.zsize.y * di.zsize.z;
        alloc_bytes = sizeof(F) * std::max(std::max(lx, ly), lz);

        ptr = (F*)std::malloc(alloc_bytes);
        if(ptr == NULL) ERROR("malloc failed");
        int lock = mlock(ptr, alloc_bytes);
        if(lock) fprintf(stderr, "memory region cannot be pinned\n");
        memset(ptr, 0, alloc_bytes);
    }
    ~Array(){
        //std::free(ptr);
    }
    void save(std::string fn){
        save_array(ptr, sizeof(F), di.decomp_info_fortran_index, pt, (void*)fn.c_str(), fn.size());
    }
    void over(std::function<void (const int&, const int&, const int&, F&)> closure){
        F* ptr2 = ptr;
        int ix, iy, iz;
        int ixg, iyg, izg;
        int ixst, iyst, izst;
        int xsz, ysz, zsz;
        switch(pt){
            case XD:
                ixst = di.xstart.x; iyst = di.xstart.y; izst = di.xstart.z;
                xsz = di.xsize.x; ysz = di.xsize.y; zsz = di.xsize.z;
                break;
            case YD:
                ixst = di.ystart.x; iyst = di.ystart.y; izst = di.ystart.z;
                xsz = di.ysize.x; ysz = di.ysize.y; zsz = di.ysize.z;
                break;
            case ZD:
                ixst = di.zstart.x; iyst = di.zstart.y; izst = di.zstart.z;
                xsz = di.zsize.x; ysz = di.zsize.y; zsz = di.zsize.z;
                break;
            default:
                ERROR("cannot happen");
        }
        for(iz = 0; iz < zsz; iz++){
            izg = izst + iz;
            for(iy = 0; iy < ysz; iy++){
                iyg = iyst + iy;
                for(ix = 0; ix < xsz; ix++){
                    ixg = ixst + ix;
                    closure(ixg, iyg, izg, *ptr2);
                    ptr2++; }}}
    }
    void transpose_into(Array<F>& other){
        if(di.decomp_info_fortran_index != other.di.decomp_info_fortran_index)
            ERROR("arrays belong to different decomp_info objects, not transposeable")
        global_transposition(ptr, pt, other.ptr, other.pt, di.decomp_info_fortran_index);
    }
};

template <typename RT>
class DistributedFFT {
private:
    typedef std::complex<RT> CT;
    cl::Context& context;
    cl::CommandQueue& queue;
    const DecompInfo di_real, di_cmpl;
    clfftPlanHandle plan_x_r2c, plan_x_c2r, plan_y, plan_z;
    Array<CT> interm;
public:
    ~DistributedFFT(){
        //clfftDestroyPlan(&plan_x_r2c);
        //clfftDestroyPlan(&plan_x_c2r);
        //clfftDestroyPlan(&plan_y);
        //clfftDestroyPlan(&plan_z);
    }
    DistributedFFT(cl::Context& _context, cl::CommandQueue& _queue, const DecompInfo& _di_real, const DecompInfo& _di_cmpl):
        context(_context), queue(_queue), di_real(_di_real), di_cmpl(_di_cmpl), interm(_di_cmpl){

        size_t shape[3] = {};
        // plan_x_r2c: INPUT REAL X-stencil, OUTPUT CMPL X-stencil
        shape = {di_real.xsize.x, di_real.xsize.y, di_real.xsize.z};
        OCLERR(clfftCreateDefaultPlan(&plan_x_r2c, context.object_, CLFFT_1D, shape));
        // STRIDES
        shape = {1, di_real.xsize.x, di_real.xsize.x * di_real.xsize.y};
        OCLERR(clfftSetPlanInStride(plan_x_r2c, CLFFT_3D, shape));
        shape = {1, di_cmpl.xsize.x, di_cmpl.xsize.x * di_cmpl.xsize.y};
        OCLERR(clfftSetPlanOutStride(plan_x_r2c, CLFFT_3D, shape));

        // plan_x_c2r: INPUT CMPL X-stencil, OUTPUT REAL X-stencil
        shape = {di_cmpl.xsize.x, di_cmpl.xsize.y, di_cmpl.xsize.z};
        OCLERR(clfftCreateDefaultPlan(&plan_x_c2r, context.object_, CLFFT_1D, shape));
        // STRIDES
        shape = {1, di_cmpl.xsize.x, di_cmpl.xsize.x * di_cmpl.xsize.y};
        OCLERR(clfftSetPlanInStride(plan_x_c2r, CLFFT_3D, shape));
        shape = {1, di_real.xsize.x, di_real.xsize.x * di_real.xsize.y};
        OCLERR(clfftSetPlanOutStride(plan_x_c2r, CLFFT_3D, shape));

        // plan_y: INPUT CMPL Y-stencil, OUTPUT CMPL Y-stencil - FFT along dim Y
        shape = {di_cmpl.ysize.y, di_cmpl.ysize.x, di_cmpl.ysize.z};
        OCLERR(clfftCreateDefaultPlan(&plan_y,     context.object_, CLFFT_1D, shape));
        // STRIDES
        shape = {di_cmpl.ysize.x, 1, di_cmpl.ysize.x * di_cmpl.ysize.y};
        OCLERR(clfftSetPlanInStride(plan_y,  CLFFT_3D, shape));
        OCLERR(clfftSetPlanOutStride(plan_y, CLFFT_3D, shape));

        // plan_z: INPUT CMPL Z-stencil, OUTPUT CMPL Z-stencil
        shape = {di_cmpl.zsize.z, di_cmpl.zsize.x, di_cmpl.zsize.y};
        OCLERR(clfftCreateDefaultPlan(&plan_z,     context.object_, CLFFT_1D, shape));
        // STRIDES
        shape = {di_cmpl.zsize.x * di_cmpl.zsize.y, 1, di_cmpl.zsize.x};
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

        OCLERR(clfftBakePlan(plan_x_r2c, 1, &queue.object_, NULL, NULL));
        OCLERR(clfftBakePlan(plan_x_c2r, 1, &queue.object_, NULL, NULL));
        OCLERR(clfftBakePlan(plan_y,     1, &queue.object_, NULL, NULL));
        OCLERR(clfftBakePlan(plan_z,     1, &queue.object_, NULL, NULL));
    }
    void execute_x_r2c(cl_mem* in, cl_mem* out){
        OCLERR(clfftEnqueueTransform(plan_x_r2c, CLFFT_FORWARD, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        queue.finish();
    }
    void execute_x_c2r(cl_mem* in, cl_mem* out){
        OCLERR(clfftEnqueueTransform(plan_x_c2r, CLFFT_BACKWARD, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        queue.finish();
    }
    void execute_y(cl_mem* in, cl_mem* out, clfftDirection dir){
        OCLERR(clfftEnqueueTransform(plan_y, dir, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        queue.finish();
    }
    void execute_z(cl_mem* in, cl_mem* out, clfftDirection dir){
        OCLERR(clfftEnqueueTransform(plan_z, dir, 1, &queue.object_, 0, NULL, NULL, in, out, NULL));
        queue.finish();
    }
    void r2c(Array<RT>& in, Array<CT>& out){
        RT* ptr_in = in.ptr;
        CT* ptr_out = out.ptr;
        CT* ptr_interm = interm.ptr;
        cl::Buffer buff_in(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, in.alloc_bytes, ptr_in);
        cl::Buffer buff_out(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, out.alloc_bytes, ptr_out);
        cl::Buffer buff_interm(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, out.alloc_bytes, ptr_interm);
        if((void*)ptr_in == (void*)ptr_out) ERROR("in-place transforms are not supported");
        if(in.pt != Array<RT>::XD) ERROR("wrong decomposition on input");
        if(out.pt != Array<CT>::ZD) ERROR("wrong decomposition on input");

        // x - fft
        //execute_x_r2c(&buff_in.object_, &buff_out.object_);
        // x -> y
        interm.set_y_pencil();
        out.set_x_pencil();
        out.transpose_into(interm);
        // y - fft
        //execute_y(&buff_interm.object_, &buff_out.object_, CLFFT_FORWARD);
        // y -> z
        out.set_y_pencil();
        interm.set_z_pencil();
        out.transpose_into(interm);
        // z - fft
        //execute_z(&buff_interm.object_, &buff_out.object_, CLFFT_FORWARD);
        out.set_z_pencil();
    }
    void c2r(Array<CT>& in, Array<RT>& out){
        CT* ptr_in = in.ptr;
        RT* ptr_out = out.ptr;
        CT* ptr_interm = interm.ptr;
        cl::Buffer buff_in(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, in.alloc_bytes, ptr_in);
        cl::Buffer buff_out(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, out.alloc_bytes, ptr_out);
        cl::Buffer buff_interm(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, out.alloc_bytes, ptr_interm);
        if((void*)ptr_in == (void*)ptr_out) ERROR("in-place transforms are not supported");
        if(in.pt != Array<CT>::ZD) ERROR("wrong decomposition on input");
        if(out.pt != Array<RT>::XD) ERROR("wrong decomposition on input");

        // z - ifft
        //execute_z(&buff_in.object_, &buff_interm.object_, CLFFT_BACKWARD);
        // z -> y
        interm.set_z_pencil();
        in.set_y_pencil();
        interm.transpose_into(in);
        // y - ifft
        //execute_y(&buff_in.object_, &buff_interm.object_, CLFFT_BACKWARD);
        // y -> x
        interm.set_y_pencil();
        in.set_x_pencil();
        interm.transpose_into(in);
        // x - ifft
        //execute_x_c2r(&buff_in.object_, &buff_out.object_);
        in.set_z_pencil();
    }
};

#ifdef SINGLEFLOAT
typedef float Float;
#else
typedef double Float;
#endif
typedef std::complex<Float> Complex;

void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi0; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi1; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi2; }
void init_cmpl_array(const int& gi0, const int& gi1, const int& gi2, std::complex<Float>& v){ v = {Float(gi0), Float(gi1)}; }

int3 to_hermitian(int3 v){ return {v.x/2+1, v.y, v.z}; }

int main(int argc, char* argv[]){
    int3 real_shape = {8, 8, 8};
    int3 cmpl_shape = to_hermitian(real_shape);

    Bookkeeping<Float> bk(0, CL_DEVICE_TYPE_CPU, real_shape);

    DecompInfo decomp_real(real_shape);
    DecompInfo decomp_cmpl(cmpl_shape);

    std::cout << decomp_real << std::endl;
    std::cout << decomp_cmpl << std::endl;

    Array<Float> arr1(decomp_real);
    Array<Complex> arr2(decomp_cmpl);
    arr2.set_z_pencil();

    arr1.over(init_array);
    arr2.over(init_cmpl_array);

    arr1.save("real.bin");
    arr2.save("cmpl.bin");

    DistributedFFT<Float> fft(bk.context, bk.queue, decomp_real, decomp_cmpl);

    fft.r2c(arr1, arr2);
    fft.c2r(arr2, arr1);

    std::cerr << "program terminating" << std::endl;

    return EXIT_SUCCESS;
}
