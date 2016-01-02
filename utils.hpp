#pragma once
#include "timing.hpp"

#define ERROR(msg) { std::fprintf(stderr, "ERR @ %s:%d > %s\n", __FILE__, __LINE__, msg); std::exit(EXIT_FAILURE); }

struct int3 {
    typedef int* IPTR;
    int x,y,z;
    operator IPTR(){ return IPTR(this); }
} __attribute__((packed));
std::ostream& operator<<(std::ostream& os, const int3& i3){
    os << "(" << i3.x << "," << i3.y << "," << i3.z << ")";
    return os;
}
bool operator==(int3& a, int3& b){ return a.x==b.x and a.y==b.y and a.z==b.z; }
int prod(int3 i3){ return i3.x * i3.y * i3.z; }
void int3_to_shape(size_t shape[3], int3 i3){ shape[0] = i3.x; shape[1] = i3.y; shape[2] = i3.z; }
int3 to_hermitian(int3 v){ return {v.x/2+1, v.y, v.z}; }
