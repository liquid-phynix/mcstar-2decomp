#pragma once

namespace DFFT {
    template<typename F, typename A> void r2c(F& f, A& a, A& b){
        if(not (a.is_real() and b.is_cmpl() and a.is_x() and b.is_z())) ERROR("sanity check failed in r2c()");
        // STEP 1
        b.as_x();
        f.forward(a, b); // x-fft, real -> cmpl
        // STEP 2
        a.as_cmpl(); a.as_y();
        //std::cerr << "from " << b << " to " << a << std::endl;
        b >> a; // x -> y
        // STEP 3
        b.as_y();
        f.forward(a, b); // y-fft, cmpl -> cmpl
        // STEP 4
        a.as_z();
        //std::cerr << "from " << b << " to " << a << std::endl;
        b >> a; // y -> z
        // STEP 5
        b.as_z();
        f.forward(a, b); // z-fft, cmpl -> cmpl
        a.as_real();
        a.as_x();
    }
    template<typename F, typename A> void c2r(F& f, A& a, A& b){
        if(not (a.is_cmpl() and b.is_real() and a.is_z() and b.is_x())) ERROR("sanity check failed in c2r()");
        // STEP 1
        b.as_cmpl(); b.as_z();
        f.backward(a, b); // z-ifft, cmpl -> cmpl
        // STEP 2
        a.as_y();
        //std::cerr << "from " << b << " to " << a << std::endl;
        b >> a; // z -> y
        // STEP 3
        b.as_y();
        f.backward(a, b); // y-ifft, cmpl -> cmpl
        // STEP 4
        a.as_x();
        //std::cerr << "from " << b << " to " << a << std::endl;
        b >> a; // y -> x
        // STEP 5
        b.as_real(); b.as_x();
        f.backward(a, b); // x-ifft, cmpl -> real
        a.as_z();
    }
};
