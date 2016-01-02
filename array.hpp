#pragma once
#include "utils.hpp"
#include "decomp.hpp"

namespace ArrayDecomp {
    class ArrayDecomp {
        enum DecompType {XD = 1, YD, ZD};
        enum AccessType { RA = 1, CA };
    public:
        DecompType dectype;
        AccessType acctype;
        const DecompInfo realdec, cmpldec;
        ArrayDecomp(int3 shape) : dectype(XD), acctype(RA), realdec(DecompInfo(shape)), cmpldec(DecompInfo({shape.x/2+1,shape.y,shape.z})){
            // sanity check: for the R2C and C2R transforms, the decomposition is x-complete
            // and along the y and z dimensions the real and cmpl decompositions mus tbe the same
            bool sane = realdec.xsize.y ==cmpldec.xsize.y  and realdec.xsize.z ==cmpldec.xsize.z;
            sane     *= realdec.xstart.y==cmpldec.xstart.y and realdec.xstart.z==cmpldec.xstart.z;
            sane     *= realdec.xend.y  ==cmpldec.xend.y   and realdec.xend.z  ==cmpldec.xend.z;
            if(not sane) ERROR("real and cmpl decompositions cannot work together");
        }
        ArrayDecomp(const ArrayDecomp& from) : dectype(from.dectype), acctype(from.acctype), realdec(from.realdec), cmpldec(from.cmpldec){}
        bool is_real(){ return acctype == RA; }
        bool is_cmpl(){ return acctype == CA; }
        bool is_x(){ return dectype == XD; }
        bool is_y(){ return dectype == YD; }
        bool is_z(){ return dectype == ZD; }
        void as_real(){ acctype = RA; }
        void as_cmpl(){ acctype = CA; }
        void as_x(){ dectype = XD; }
        void as_y(){ dectype = YD; }
        void as_z(){ dectype = ZD; }
    protected:
        void save(const char* fn, size_t len, void* source, size_t float_size){
            save_array(source, (acctype==RA ? 1 : 2) * float_size,
                    acctype==RA ? realdec.get_index() : cmpldec.get_index(),
                    dectype, (void*)fn, len);
        }
    };


    /*
    template <typename T>
    void over(std::function<void (const int&, const int&, const int&, T&)> closure){

        F* ptr2 = ptr;
        int ix, iy, iz, ixg, iyg, izg, ixst, iyst, izst, xsz, ysz, zsz;
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
            default: ERROR("cannot happen"); }
                     for(iz = 0; iz < zsz; iz++){
                         izg = izst + iz;
                         for(iy = 0; iy < ysz; iy++){
                             iyg = iyst + iy;
                             for(ix = 0; ix < xsz; ix++){
                                 ixg = ixst + ix;
                                 closure(ixg, iyg, izg, *ptr2);
                                 ptr2++; }}}
    }
    */
}
