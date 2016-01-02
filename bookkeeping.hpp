#pragma once
#include <mpi.h>
#include "interop.h"

template<typename RT>
class Bookkeeping {
private:
    int size, rank;
    int id_within_node, node_size;
public:
    int get_size(){ return size; }
    int get_rank(){ return rank; }
    int get_id_within_node(){ return id_within_node; }
    int get_node_size(){ return node_size; }
    Bookkeeping(int3 gshape){
        MPI::Init();
        size = MPI::COMM_WORLD.Get_size();
        rank = MPI::COMM_WORLD.Get_rank();
        char pname[MPI_MAX_PROCESSOR_NAME] = {};
        int pnamelen;
        MPI::Get_processor_name(pname, pnamelen);

        char* othernames = NULL;
        if(rank == 0) othernames = new char[size * MPI_MAX_PROCESSOR_NAME]{};

        MPI::COMM_WORLD.Gather(pname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, othernames, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, 0);

        int* id_by_rank{};
        int* clique_size{};
        if(rank == 0){
            //for(int i = 0; i < size; i++)
                //std::cerr << "> rank " << i << " is named |" << &othernames[i * MPI_MAX_PROCESSOR_NAME] << "|" << std::endl;
            id_by_rank = new int[size];
            int _pairid[size];
            int* pairid_ptr[size];
            int* pairid = _pairid;
            for(int id = 0; id < size; id++) { id_by_rank[id] = -1; }
            for(int id = 0; id < size; id++){
                if(id_by_rank[id] != -1) continue; // pair already found
                //int pairid = 0; // id reset
                *pairid = 0;
                id_by_rank[id] = *pairid;
                pairid_ptr[id] = pairid;
                for(int other = 0; other < size; other++){
                    if(id_by_rank[other] != -1) continue;
                    if(0 == strncmp(&othernames[id * MPI_MAX_PROCESSOR_NAME],
                                    &othernames[other * MPI_MAX_PROCESSOR_NAME],
                                    MPI_MAX_PROCESSOR_NAME)){
                        *pairid += 1;
                        id_by_rank[other] = *pairid;
                        pairid_ptr[other] = pairid;
                    }
                }
                pairid++;
            }
            clique_size = new int[size];
            for(int id = 0; id < size; id++) { clique_size[id] = *pairid_ptr[id]; }
        }
        MPI::COMM_WORLD.Scatter(id_by_rank, 1, MPI_INT, &id_within_node, 1, MPI_INT, 0);
        MPI::COMM_WORLD.Scatter(clique_size, 1, MPI_INT, &node_size, 1, MPI_INT, 0);
        delete[] id_by_rank;
        delete[] clique_size;
        delete[] othernames;
        std::cerr << "rank (" << rank << ") has id (" << id_within_node << ") within clique of size (" << node_size << ")" << std::endl;

        int proc_grid[2] = {0, 0};
        initialize(gshape, proc_grid);
        int float_size;
        get_float_size(&float_size);
        if(float_size != sizeof(RT)){
            fprintf(stderr, "sizeof(F) = %d , mytype_bytes = %d\n", sizeof(RT), float_size);
            ERROR("main() and 2DECOMP float sizes differ"); }
    }
    bool check_if_homogeneous(char* buffer, int len){
        int* lens{};
        if(rank == 0) lens = new int[size];
        MPI::COMM_WORLD.Gather(&len, 1, MPI_INT, lens, 1, MPI_INT, 0);
        for(int id = 0; id < size; id++){
            if(lens[id] != len) ERROR("len parameter in check_if_homogeneous is non-homogeneous");
        }
        delete[] lens;

        char* bigstring{};
        if(rank == 0) bigstring = new char[size * len];
        MPI::COMM_WORLD.Gather(buffer, len, MPI_BYTE, bigstring, len, MPI_BYTE, 0);
        if(rank == 0){
            bool check = true;
            for(int id = 0; id < size; id++){
                if(0 != strncmp(buffer, bigstring + id * len, len)){
                    check = false;
                    break;
                }
            }
            for(int id = 0; id < size; id++){ lens[id] = check; }
        }
        delete[] bigstring;
        int ret;
        MPI::COMM_WORLD.Scatter(lens, 1, MPI_INT, &ret, 1, MPI_INT, 0);
        return ret;
    }
    ~Bookkeeping(){ finalize(); MPI::Finalize(); }
};
