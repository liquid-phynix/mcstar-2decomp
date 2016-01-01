namespace FFTW {

#ifdef SINGLEFLOAT
#define FPREF(a) fftwf_##a
#else
#define FPREF(a) fftw_##a
#endif

#include "utils.hpp"
#include "decomp.hpp"
#include <fftw3.h>
    template <typename F> class FFT {
    protected:
        typedef F                RT;
        typedef std::complex<RT> CT;
    private:
        FPREF(plan) plan_x_r2c, plan_x_c2r, plan_y_forw, plan_y_back, plan_z_forw, plan_z_back;
    public:
        FFT(DecompInfo real, DecompInfo cmpl, RT* real_ptr, CT* _cmpl_ptr, CT* _cmpl_ptr_im){
            if(sizeof(FPREF(complex)) != sizeof(CT)) ERROR("main() and fftw float types differ");
            FPREF(complex)* cmpl_ptr = reinterpret_cast<FPREF(complex)*>(_cmpl_ptr);
            FPREF(complex)* cmpl_ptr_im = reinterpret_cast<FPREF(complex)*>(_cmpl_ptr_im);
            FPREF(iodim) transform, repeat[3];
            int transform_rank, repeat_rank; int real_array_size[3]; int array_size[3];

            // sanity check: for the R2C and C2R cases the x-decomposition sizes and start/end indices
            // must be the same for the real and complex decompositions
            bool sane = real.xsize.y==cmpl.xsize.y and real.xsize.z==cmpl.xsize.z;
            sane     *= real.xstart.y==cmpl.xstart.y and real.xstart.z==cmpl.xstart.z;
            sane     *= real.xend.y==cmpl.xend.y and real.xend.z==cmpl.xend.z;
            if(not sane) ERROR("real and cmpl decompositions cannot work together");

            // plan for x-directional decomposition
            // xsize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            real_array_size[0] = real.xsize.z; real_array_size[1] = real.xsize.y; real_array_size[2] = real.xsize.x;
            array_size[0] = cmpl.xsize.z; array_size[1] = cmpl.xsize.y; array_size[2] = cmpl.xsize.x;

            transform_rank = 1;          // transform dimension
            transform.n = array_size[2]; // transform length
            transform.is = transform.os = 1;
            repeat_rank = 2;             // 2D repetition
            repeat[0].n = array_size[0]; repeat[0].is = real_array_size[1] * real_array_size[2]; repeat[0].os = array_size[1] * array_size[2];
            repeat[1].n = array_size[1]; repeat[1].is = real_array_size[2]; repeat[1].os = array_size[2];
            plan_x_r2c = FPREF(plan_guru_dft_r2c)(transform_rank, &transform, repeat_rank, repeat, real_ptr, cmpl_ptr, FFTW_ESTIMATE);
            plan_x_c2r = FPREF(plan_guru_dft_c2r)(transform_rank, &transform, repeat_rank, repeat, cmpl_ptr, real_ptr, FFTW_ESTIMATE);

            // plan for y-directional decomposition
            // ysize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            array_size[0] = cmpl.ysize.z; array_size[1] = cmpl.ysize.y; array_size[2] = cmpl.ysize.x;
            transform_rank = 1;          // transform dimension
            transform.n = array_size[1]; // transform length
            transform.is = transform.os = array_size[2];
            repeat_rank = 2;             // 2D repetition
            repeat[0].n = array_size[0]; repeat[0].is = repeat[0].os = array_size[2] * array_size[1];
            repeat[1].n = array_size[2]; repeat[1].is = repeat[1].os = 1;
            plan_y_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, cmpl_ptr_im, cmpl_ptr, FFTW_FORWARD, FFTW_ESTIMATE);
            plan_y_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, cmpl_ptr, cmpl_ptr_im, FFTW_BACKWARD, FFTW_ESTIMATE);

            // plan for z-directional decomposition
            // zsize describes the array, extent of dimensions is reversed
            // so that {slow, medium, fast} follows C convention
            array_size[0] = cmpl.zsize.z; array_size[1] = cmpl.zsize.y; array_size[2] = cmpl.zsize.x;
            transform_rank = 1;          // transform dimension
            transform.n = array_size[0]; // transform length
            transform.is = transform.os = array_size[1] * array_size[2];
            repeat_rank = 2;             // 2D repetition
            repeat[0].n = array_size[1]; repeat[0].is = repeat[0].os = array_size[2];
            repeat[1].n = array_size[2]; repeat[1].is = repeat[1].os = 1;
            plan_z_forw = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, cmpl_ptr_im, cmpl_ptr, FFTW_FORWARD, FFTW_ESTIMATE);
            plan_z_back = FPREF(plan_guru_dft)(transform_rank, &transform, repeat_rank, repeat, cmpl_ptr, cmpl_ptr_im, FFTW_BACKWARD, FFTW_ESTIMATE);
        }
        ~FFT(){
            FPREF(destroy_plan)(plan_x_r2c);
            FPREF(destroy_plan)(plan_x_c2r);
            FPREF(destroy_plan)(plan_y_forw);
            FPREF(destroy_plan)(plan_y_back);
            FPREF(destroy_plan)(plan_z_forw);
            FPREF(destroy_plan)(plan_z_back);
        }
        void execute_x_r2c(){ FPREF(execute(plan_x_r2c)); }
        void execute_x_c2r(){ FPREF(execute(plan_x_c2r)); }
        void execute_y_f(){ FPREF(execute(plan_y_forw)); }
        void execute_y_b(){ FPREF(execute(plan_y_back)); }
        void execute_z_f(){ FPREF(execute(plan_z_forw)); }
        void execute_z_b(){ FPREF(execute(plan_z_back)); }
    };
}
