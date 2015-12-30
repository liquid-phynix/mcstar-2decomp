#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <list>
#include <utility>
#include <string>
#include <functional>
#include <mpi.h>
#include <unistd.h>
#include <sys/mman.h>

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

class Bookkeeping {
    public:
        static const cl_device_type ALL = CL_DEVICE_TYPE_ALL, GPU = CL_DEVICE_TYPE_GPU, CPU = CL_DEVICE_TYPE_CPU;
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
            ERROR("devicenum out of range");
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
    }
    ~Bookkeeping(){
        finalize();
        OCLERR(clfftTeardown());
        MPI::Finalize();
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
private:
    const DecompInfo di;
    F* ptr;
    int alloc_bytes;
public:
    Array(DecompInfo _di, PencilType _pt = XD): di(_di), ptr(NULL), alloc_bytes(0), pt(_pt){
        if(di.float_size != sizeof(F)){
            fprintf(stderr, "sizeof(F) = %d , mytype_bytes = %d\n", sizeof(F), di.float_size);
            ERROR("main() and 2DECOMP float sizes differ");
        }
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
        std::free(ptr);
    }
    void save_real(std::string fn){
        int fn_len = fn.size();
        int elem_size = sizeof(F);
        save_array(ptr, &elem_size, &di.decomp_info_fortran_index, &pt, fn.c_str(), &fn_len);
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
};

/*
    void r2c(std::string in, std::string out){
        void* in_ptr = get_array(in);
        void* out_ptr = get_array(out);
        void* im_ptr = get_array("intermediate");
        int ttype = 0;

        // x - fft
        //

        // x -> y
        ttype = 1;
        __iop_MOD_iop_transpose(&ttype, &in_ptr, xsize, &out_ptr, ysize);

        // y - fft
        //

        // y -> z
        ttype = 2;
        __iop_MOD_iop_transpose(&ttype, &in_ptr, xsize, &out_ptr, ysize);

        // z - fft
        //
    }
    void c2r(std::string in, std::string out){
        void* in_ptr = get_array(in);
        void* out_ptr = get_array(out);
        void* im_ptr = get_array("intermediate");
        int ttype = 0;

        // z - ifft
        //

        // z -> y
        ttype = 3;
        __iop_MOD_iop_transpose(&ttype, &in_ptr, xsize, &out_ptr, ysize);

        // y - ifft
        //

        // y -> x
        ttype = 4;
        __iop_MOD_iop_transpose(&ttype, &in_ptr, xsize, &out_ptr, ysize);

        // x - ifft
        //
    }
};
*/

#ifdef SINGLEFLOAT
typedef float Float;
//typedef std::complex<float> Complex;
#else
typedef double Float;
//typedef std::complex<double> Complex;
#endif

void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi0; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi1; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi2; }
void init_cmpl_array(const int& gi0, const int& gi1, const int& gi2, std::complex<Float>& v){ v = {Float(gi0), Float(gi1)}; }

int main(int argc, char* argv[]){
    int3 real_shape = {8, 8, 8};
    int3 cmpl_shape = {real_shape.x / 2 + 1, real_shape.y, real_shape.z};

    Bookkeeping(0, Bookkeeping::CPU, real_shape);

    DecompInfo decomp_real(real_shape);
    std::cout << decomp_real << std::endl;
    DecompInfo decomp_cmpl(cmpl_shape);
    std::cout << decomp_cmpl << std::endl;

    Array<Float> arr(decomp_real);

/*
    DistributedFFT<Float> fft(8, 8, 8);
    fft.add_array("real_arr");
    fft.over_real("real_arr", init_array);
    fft.save_real("real_arr", "rout.bin");

    fft.add_array("cmpl_arr");
    fft.add_array("cmpl_arr_2");
    fft.over_cmpl("cmpl_arr", init_cmpl_array);
    fft.save_cmpl("cmpl_arr", "cout.bin", 1);

    void* in_ptr = fft.get_array("cmpl_arr");
    void* out_ptr = fft.get_array("cmpl_arr_2");
    //void* im_ptr = fft.get_array("intermediate");
    int ttype = 0;

    ttype = 1;
    __iop_MOD_iop_transpose(&ttype, &in_ptr, fft.xsize, &out_ptr, fft.ysize);
    //ttype = 4;
    //__iop_MOD_iop_transpose(&ttype, &out_ptr, fft.ysize, &in_ptr, fft.xsize);
    //fft.save_cmpl("cmpl_arr", "c2out.bin", 1);
    fft.save_cmpl("cmpl_arr_2", "c2out.bin", 2);


    //fft.r2c("real_arr", "cmpl_arr");
    //fft.c2r("cmpl_arr", "real_arr");
    */

    return EXIT_SUCCESS;
}
