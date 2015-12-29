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
    void __iop_MOD_iop_transpose(int*, void**, int*, void**, int*);
    // pencil_kind: x -> 1, y -> 2, z -> 3
    void __iop_MOD_save_real(int*, void**, int*, const char**, int*);
    void __iop_MOD_save_cmpl(int*, void**, int*, const char**, int*);
}

#define ERROR(msg) { std::fprintf(stderr, "ERR @ %s:%d> %s\n", __FILE__, __LINE__, msg); std::exit(1); }

// real global: nx*ny*nz
// cmpl global: (nx//2+1)*ny*nz
// real: x-stencil
// cmpl: z-stencil

static int _DistributedFFT = 0;
template <typename F>
class DistributedFFT {
//private:
public:
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
        printf("rank %d\n", MPI::COMM_WORLD.Get_rank());
        printf("x-pencil: (%d-%d)/(%d-%d)/(%d-%d)\n", xstart[0], xend[0], xstart[1], xend[1], xstart[2], xend[2]);
        printf("y-pencil: (%d-%d)/(%d-%d)/(%d-%d)\n", ystart[0], yend[0], ystart[1], yend[1], ystart[2], yend[2]);
        printf("z-pencil: (%d-%d)/(%d-%d)/(%d-%d)\n\n", zstart[0], zend[0], zstart[1], zend[1], zstart[2], zend[2]);
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
        memset(ptr, 0, general_allocation_size);
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
    void over_real(std::string name, std::function<void (const int&, const int&, const int&, F&)> closure){
        F* ptr = NULL;
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++)
            if(it->first == name) ptr = reinterpret_cast<F*>(it->second);
        if(ptr == NULL) ERROR("array name not found in list");
        int i2g, i1g, i0g;
        for(int i2 = 0; i2 < xsize[2]; i2++){
            i2g = xstart[2] + i2;
            for(int i1 = 0; i1 < xsize[1]; i1++){
                i1g = xstart[1] + i1;
                for(int i0 = 0; i0 < rnx; i0++){
                    closure(i0, i1g, i2g, *ptr);
                    ptr++;
                }
            }
        }
    }
    void over_cmpl(std::string name, std::function<void (int&, int&, int&, std::complex<F>&)> closure){
        std::complex<F>* ptr = NULL;
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++)
            if(it->first == name) ptr = reinterpret_cast<std::complex<F>*>(it->second);
        if(ptr == NULL) ERROR("array name not found in list");
        int i2g, i1g, i0g;
        for(int i2 = 0; i2 < zsize[2]; i2++){
            i2g = zstart[2] + i2;
            for(int i1 = 0; i1 < zsize[1]; i1++){
                i1g = zstart[1] + i1;
                for(int i0 = 0; i0 < zsize[0]; i0++){
                    i0g = zstart[0] + i0;
                    closure(i0g, i1g, i2g, *ptr);
                    ptr++;
                }
            }
        }
    }
    void save_real(std::string name, std::string fn, int pencil_kind = 1){
        void* ptr = NULL;
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++)
            if(it->first == name) ptr = it->second;
        if(ptr == NULL) ERROR("array name not found in list");
        int len = fn.size();
        const char* str = fn.c_str();
        int globalshape[3] = {rnx, ny, nz};
        __iop_MOD_save_real(&pencil_kind, &ptr, globalshape, &str, &len);
    }
    void save_cmpl(std::string name, std::string fn, int pencil_kind = 3){
        void* ptr = NULL;
        for(AList::iterator it = m_arrays.begin(); it != m_arrays.end(); it++)
            if(it->first == name) ptr = it->second;
        if(ptr == NULL) ERROR("array name not found in list");
        int len = fn.size();
        const char* str = fn.c_str();
        int globalshape[3] = {cnx, ny, nz};
        __iop_MOD_save_cmpl(&pencil_kind, &ptr, globalshape, &str, &len);
    }
};

void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi0; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi1; }
//void init_array(const int& gi0, const int& gi1, const int& gi2, Float& v){ v = gi2; }
void init_cmpl_array(const int& gi0, const int& gi1, const int& gi2, std::complex<Float>& v){ v = {Float(gi0), Float(gi1)}; }

int main(int argc, char* argv[]){

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

    return 0;
}
