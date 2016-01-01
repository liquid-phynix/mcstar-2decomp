#pragma once

extern "C" {
    void get_float_size(int*);
    void initialize(int*, int*);
    void finalize();
    void create_decomp_info(int*, int*, int*, int*);
    void save_array(void*, int, int, int, void*, int);
    void global_transposition(void*, int, void*, int, int);
}
