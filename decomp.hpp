#pragma once
#include "interop.h"

struct DecompInfo {
    int decomp_info_fortran_index, decomp_info_rank;
    int3 xstart, xend, xsize, ystart, yend, ysize, zstart, zend, zsize;
    DecompInfo() = delete;
    DecompInfo(const DecompInfo& other) = default;
    DecompInfo(int3 shape){
        int3 ret[3 * 3] = {};
        if(shape.x <= 1 || shape.y <= 1 || shape.z <= 1)
            ERROR("dimensions must be > 1");
        create_decomp_info(shape, &decomp_info_fortran_index, &decomp_info_rank, &ret[0].x);
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
