namespace FFTW {

#ifdef SINGLEFLOAT
#define FPREF(a) fftwf_##a
#else
#define FPREF(a) fftw_##a
#endif

#include <complex>
#include "utils.hpp"
#include <fftw3.h>
//#include <cstdlib>
#include <string>
#include <sys/mman.h>

    template <typename RT, class Base>
    class Array : public Base {
        const int3 shape;
        size_t alloc_len, alloc_bytes;
        void* ptr;
    public:
        Array(const Array& from) : Base(from), shape(from.shape){ init(); }
        Array(int3 _shape) : Base(_shape), shape(_shape){ init(); }
        void init(){
            alloc_len = 2 * std::max(std::max(prod(this->cmpldec.xsize), prod(this->cmpldec.ysize)), prod(this->cmpldec.zsize));
            alloc_bytes = sizeof(RT) * alloc_len;
            ptr = FPREF(malloc)(alloc_bytes);
            int lock = mlock(ptr, alloc_bytes);
            if(lock) fprintf(stderr, "memory region cannot be pinned\n");
        }
        ~Array(){ FPREF(free)(ptr); }
        void save(std::string fn){ save(fn.c_str(), fn.size(), ptr, sizeof(RT)); };
        RT* real_ptr(){ return reinterpret_cast<RT*>(ptr); }
        FPREF(complex)* cmpl_ptr(){ return reinterpret_cast<FPREF(complex)*>(ptr); }
    };

    template <typename F, class Arr> class FFT {
    private:
        FPREF(plan) plan_x_r2c, plan_x_c2r, plan_y_forw, plan_y_back, plan_z_forw, plan_z_back;
    public:
        typedef F                RT;
        typedef std::complex<RT> CT;
        FFT(Arr& scratchA, Arr& scratchB){
            if(sizeof(FPREF(complex)) != sizeof(CT)) ERROR("main() and fftw float types differ");
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
            plan_x_r2c = FPREF(plan_guru_dft_r2c)(transform_rank, &transform, repeat_rank, repeat, scratchA.real_ptr(), scratchB.cmpl_ptr(), FFTW_DESTROY_INPUT | FFTW_MEASURE);

            transform.n = array_size[2]; // transform length
            transform.is = transform.os = 1;
            repeat[0].n = array_size[0]; repeat[0].is = array_size[1] * array_size[2]; repeat[0].os = real_array_size[1] * real_array_size[2];
            repeat[1].n = array_size[1]; repeat[1].is = array_size[2]; repeat[1].os = real_array_size[2];
            plan_x_c2r = FPREF(plan_guru_dft_c2r)(transform_rank, &transform, repeat_rank, repeat, scratchA.cmpl_ptr(), scratchB.real_ptr(), FFTW_DESTROY_INPUT | FFTW_MEASURE);

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
            plan_y_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.cmpl_ptr(), scratchB.cmpl_ptr(), FFTW_FORWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            plan_y_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.cmpl_ptr(), scratchB.cmpl_ptr(), FFTW_BACKWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
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
            plan_z_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.cmpl_ptr(), scratchB.cmpl_ptr(), FFTW_FORWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
            plan_z_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, scratchA.cmpl_ptr(), scratchB.cmpl_ptr(), FFTW_BACKWARD, FFTW_DESTROY_INPUT | FFTW_MEASURE);
        }
        ~FFT(){
            FPREF(destroy_plan)(plan_x_r2c);
            FPREF(destroy_plan)(plan_x_c2r);
            FPREF(destroy_plan)(plan_y_forw);
            FPREF(destroy_plan)(plan_y_back);
            FPREF(destroy_plan)(plan_z_forw);
            FPREF(destroy_plan)(plan_z_back);
        }
        void forward(Arr& in, Arr& out){
            if(in.realdec != out.realdec or in.cmpldec != out.cmpldec)
                ERROR("arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_real() and out.is_cmpl()){
                FPREF(execute_dft_r2c(plan_x_r2c, in.real_ptr(), out.cmpl_ptr()));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_y_forw, in.cmpl_ptr(), out.cmpl_ptr()));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_z_forw, in.cmpl_ptr(), out.cmpl_ptr()));
            } else ERROR("array decomposition mismatch in FFT::forward");
        }
        void backward(Arr& in, Arr& out){
            if(in.realdec != out.realdec or in.cmpldec != out.cmpldec)
                ERROR("arrays of different decomposition index cannot be transformed");
                   if(in.is_x() and out.is_x() and in.is_cmpl() and out.is_real()){
                FPREF(execute_dft_c2r(plan_z_back, in.cmpl_ptr(), out.cmpl_ptr()));
            } else if(in.is_y() and out.is_y() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_y_back, in.cmpl_ptr(), out.cmpl_ptr()));
            } else if(in.is_z() and out.is_z() and in.is_cmpl() and out.is_cmpl()){
                FPREF(execute_dft(plan_x_c2r, in.cmpl_ptr(), out.real_ptr()));
            } else ERROR("array decomposition mismatch in FFT::backward");
        }
    };
}
