#pragma once
#include "timing.hpp"

#define ERROR(msg) { std::fprintf(stderr, "ERR @ %s:%d > %s\n", __FILE__, __LINE__, msg); std::exit(EXIT_FAILURE); }

struct int3 {
    typedef int* IPTR;
    int x,y,z;
    operator IPTR() const { return IPTR(this); }
    int prod() const { return x * y * z; }
    int3 real_to_hermitian() const { return {x/2+1, y, z}; }
    void to_shape(size_t shape[3]){ shape[0] = x; shape[1] = y; shape[2] = z; }
} __attribute__((packed));
std::ostream& operator<<(std::ostream& os, const int3& i3){
    os << "(" << i3.x << "," << i3.y << "," << i3.z << ")";
    return os;
}
bool operator==(int3& a, int3& b){ return a.x==b.x and a.y==b.y and a.z==b.z; }
