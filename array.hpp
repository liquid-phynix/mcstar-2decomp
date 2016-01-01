#pragma once
#include "decomp.hpp"
#include <sys/mman.h>

template <typename F>
class Array {
public:
    enum PencilType {XD = 1, YD, ZD};
    PencilType pt;
    size_t alloc_bytes;
    F* ptr;
    const DecompInfo di;
public:
    void set_x_pencil(){ pt = XD; }
    void set_y_pencil(){ pt = YD; }
    void set_z_pencil(){ pt = ZD; }
    Array(DecompInfo _di, PencilType _pt = XD): di(_di), ptr(NULL), alloc_bytes(0), pt(_pt){
        int elems = std::max(std::max(prod(di.xsize), prod(di.ysize)), prod(di.ysize));
        ptr = new F[elems]{};
        alloc_bytes = sizeof(F) * elems;
        int lock = mlock(ptr, alloc_bytes);
        if(lock) fprintf(stderr, "memory region cannot be pinned\n");
    }
    ~Array(){ delete[] ptr; }
    void save(std::string fn){ save_array(ptr, sizeof(F), di.decomp_info_fortran_index, pt, (void*)fn.c_str(), fn.size()); }
    void over(std::function<void (const int&, const int&, const int&, F&)> closure){
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
            default:
                ERROR("cannot happen"); }
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
            ERROR("arrays belong to different decomp_info objects, not transposeable");
        global_transposition(ptr, pt, other.ptr, other.pt, di.decomp_info_fortran_index); }
};
