#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <list>
#include <utility>
#include <string>
#include <mpi.h>
#include <sys/mman.h>

#ifdef SINGLEFLOAT
typedef float Float;
typedef std::complex<float> Complex;
#else
typedef double Float;
typedef std::complex<double> Complex;
#endif

extern "C" {
    void __decomp_2d_MOD_decomp_2d_init(int*, int*, int*, int*, int*, int*);
    void __decomp_2d_MOD_decomp_2d_finalize();
    void __iop_MOD_sync_configuration(int*, int*, int*, int*, int*, int*, int*, int*, int*);
    void __iop_MOD_get_float_size(int*);
    void __iop_MOD_iop_transpose(int*, void*, int*, void*, int*);
}

#define ERROR(msg) { std::fprintf(stderr, "ERR @ %s:%d> %s\n", __FILE__, __LINE__, msg); std::exit(1); }

// real global: nx*ny*nz
// cmpl global: (nx//2+1)*ny*nz
// real: x-stencil
// cmpl: z-stencil

static int _DistributedFFT = 0;
template <typename F>
class DistributedFFT {
private:
    typedef std::list<std::pair<std::string, void*> > AList;
    AList m_arrays;
    int rnx, cnx, ny, nz;
    int xstart[3], ystart[3], zstart[3];
    int xend[3], yend[3], zend[3];
    int xsize[3], ysize[3], zsize[3];
    int general_allocation_size;
public:
    DistributedFFT(int _nx, int _ny, int _nz){
        _DistributedFFT++;
        if(_DistributedFFT != 1)
            ERROR("only one instance of DistributedFFT supported at this time");
        general_allocation_size = 0;
        rnx = _nx; ny = _ny; nz = _nz;
        cnx = rnx / 2 + 1;
        if(_nx <= 1 || _ny <= 1 || _nz <= 1)
            ERROR("real dimensions must be >= 1");
        MPI::Init();
        int prows = 0;
        int pcols = 0;
        __decomp_2d_MOD_decomp_2d_init(&cnx, &ny, &nz, &prows, &pcols, NULL);
        int floatsize = 0;
        __iop_MOD_get_float_size(&floatsize);
        if(floatsize != sizeof(F)){
            fprintf(stderr, "sizeof(F) = %d / mytype_bytes = %d\n", sizeof(F), floatsize);
            ERROR("DistributedFFT and 2DECOMP float sizes don't match up");
        }
        __iop_MOD_sync_configuration(xstart, ystart, zstart,
                                     xend,   yend,   zend,
                                     xsize,  ysize,  zsize);
        general_allocation_size = 2 * sizeof(F) * std::max(std::max(xsize[0] * xsize[1] * xsize[2], ysize[0] * ysize[1] * ysize[2]), zsize[0] * zsize[1] * zsize[2]);
        printf("x-pencil: (%d-%d)/(%d-%d)/(%d-%d)\n", xstart[0], xend[0], xstart[1], xend[1], xstart[2], xend[2]);
        printf("y-pencil: (%d-%d)/(%d-%d)/(%d-%d)\n", ystart[0], yend[0], ystart[1], yend[1], ystart[2], yend[2]);
        printf("z-pencil: (%d-%d)/(%d-%d)/(%d-%d)\n", zstart[0], zend[0], zstart[1], zend[1], zstart[2], zend[2]);
        add_array("intermediate");
    }
    ~DistributedFFT(){
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++)
            free(it->second);
        __decomp_2d_MOD_decomp_2d_finalize();
        MPI::Finalize();
    }
    int allocation_in_mb(){
        return (m_arrays.size() + 1) * general_allocation_size / float(1024 * 1024);
    }
    void add_array(std::string name){
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++){
            if(it->first == name)
                ERROR("array name already in use");
        }
        void* ptr = std::malloc(general_allocation_size);
        if(ptr == NULL) ERROR("memory allocation failed");
        int lock = mlock(ptr, general_allocation_size);
        if(lock) fprintf(stderr, "memory region cannot be pinned\n");
        m_arrays.push_back(std::make_pair(name, ptr));
    }
    void* get_array(std::string name){
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++)
            if(it->first == name) return it->second;
        ERROR("array name not found in list");
    }
    void r2c(std::string in, std::string out){
        void* in_ptr = get_array(in);
        void* out_ptr = get_array(out);
        void* im_ptr = get_array("intermediate");
        int ttype = 0;

        // x - fft
        //

        // x -> y
        ttype = 1;
        __iop_MOD_iop_transpose(&ttype, in_ptr, xsize, out_ptr, ysize);

        // y - fft
        //

        // y -> z
        ttype = 2;
        __iop_MOD_iop_transpose(&ttype, in_ptr, xsize, out_ptr, ysize);

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
        __iop_MOD_iop_transpose(&ttype, in_ptr, xsize, out_ptr, ysize);

        // y - ifft
        //

        // y -> x
        ttype = 4;
        __iop_MOD_iop_transpose(&ttype, in_ptr, xsize, out_ptr, ysize);

        // x - ifft
        //
    }
};

int main(int argc, char* argv[]){

    DistributedFFT<Float> fft(128, 128, 128);
    fft.add_array("real_arr");
    fft.add_array("cmpl_arr");

    return 0;
}
