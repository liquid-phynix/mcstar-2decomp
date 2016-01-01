#pragma once
#include "fft-fftw.hpp"

template<class FFT>
class DistributedFFT {
    FFT fft;
    Array<typename FFT::CT>  ia;
    Array<typename FFT::RT>& ra;
    Array<typename FFT::CT>& ca;
public:
    DistributedFFT(DecompInfo rd, DecompInfo cd, Array<typename FFT::RT>& _ra, Array<typename FFT::CT>& _ca):
    fft(rd, cd), ia(cd), ra(_ra), ca(_ca){ fft.init(ra.ptr, ca.ptr, ia.ptr); }
    void r2c(){
        // x - fft
        fft.execute_x_r2c();
        // x -> y
        ia.set_y_pencil();
        ca.set_x_pencil();
        ca.transpose_into(ia);
        // y - fft
        fft.execute_y_f();
        // y -> z
        ca.set_y_pencil();
        ia.set_z_pencil();
        ca.transpose_into(ia);
        // z - fft
        fft.execute_z_f();
        ca.set_z_pencil();
    }
    void c2r(){
        // z - ifft
        fft.execute_z_b();
        // z -> y
        ia.set_z_pencil();
        ca.set_y_pencil();
        ia.transpose_into(ca);
        // y - ifft
        fft.execute_y_b();
        // y -> x
        ia.set_y_pencil();
        ca.set_x_pencil();
        ia.transpose_into(ca);
        // x - ifft
        fft.execute_x_c2r();
        ca.set_z_pencil();
    }
};
