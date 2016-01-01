#pragma once
#include "interop.h"

template<typename RT>
class Bookkeeping {
public:
    int size, rank;
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
        if(rank == 0){
            //for(int i = 0; i < size; i++)
                //std::cerr << "> rank " << i << " is named |" << &othernames[i * MPI_MAX_PROCESSOR_NAME] << "|" << std::endl;
            id_by_rank = new int[size];
            for(int id = 0; id < size; id++) { id_by_rank[id] = -1; }
            for(int id = 0; id < size; id++){
                if(id_by_rank[id] != -1) continue; // pair already found
                int pairid = 0; // id reset
                id_by_rank[id] = pairid;
                for(int other = 0; other < size; other++){
                    if(id_by_rank[other] != -1) continue;
                    if(0 == strncmp(&othernames[id * MPI_MAX_PROCESSOR_NAME],
                                    &othernames[other * MPI_MAX_PROCESSOR_NAME],
                                    MPI_MAX_PROCESSOR_NAME)){
                        id_by_rank[other] = ++pairid; }}}}
        int id_within_node{};
        MPI::COMM_WORLD.Scatter(id_by_rank, 1, MPI_INT, &id_within_node, 1, MPI_INT, 0);
        delete[] id_by_rank;
        delete[] othernames;

        int proc_grid[2] = {0, 0};
        initialize(gshape, proc_grid);
        int float_size;
        get_float_size(&float_size);
        if(float_size != sizeof(RT)){
            fprintf(stderr, "sizeof(F) = %d , mytype_bytes = %d\n", sizeof(RT), float_size);
            ERROR("main() and 2DECOMP float sizes differ"); }
    }
    ~Bookkeeping(){
        finalize();
        MPI::Finalize(); }
};
