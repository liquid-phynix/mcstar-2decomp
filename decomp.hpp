#pragma once
#include <mpi.h>
#include <sys/mman.h>
#include <string>
#include <complex>
#include "interop.h"
#include "utils.hpp"

namespace DecompImpl {

    template<typename RT> class Bookkeeping {
        int size, rank;
        int id_within_node, node_size;
    public:
        int get_size() const { return size; }
        int get_rank() const { return rank; }
        int get_id_within_node() const { return id_within_node; }
        int get_node_size() const { return node_size; }
        Bookkeeping(int3 gshape){
            MPI::Init();
            size = MPI::COMM_WORLD.Get_size();
            rank = MPI::COMM_WORLD.Get_rank();
            char pname[MPI_MAX_PROCESSOR_NAME]{};
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
                for(int id = 0; id < size; id++) { clique_size[id] = *pairid_ptr[id]+1; }
            }
            MPI::COMM_WORLD.Scatter(id_by_rank, 1, MPI_INT, &id_within_node, 1, MPI_INT, 0);
            MPI::COMM_WORLD.Scatter(clique_size, 1, MPI_INT, &node_size, 1, MPI_INT, 0);
            delete[] id_by_rank;
            delete[] clique_size;
            delete[] othernames;
            std::cerr << "rank (" << rank << ") has id (" << id_within_node << ") within clique of size (" << node_size << ")" << std::endl;

            // initialize 2DECOMP with processor grid given below, {0,0} is to autotune
            int proc_grid[2] = {0, 0};
            initialize(gshape, proc_grid);

            // checking if library float size equals to program's
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

    struct DecompInfo {
        int fi, rank;
    public:
        int get_index() const { return fi; }
        int3 xstart, xend, xsize, ystart, yend, ysize, zstart, zend, zsize;
        DecompInfo() = delete;
        DecompInfo(const DecompInfo& other) = default;
        DecompInfo(int3 shape){
            int3 ret[3 * 3] = {};
            if(shape.x <= 1 || shape.y <= 1 || shape.z <= 1) ERROR("dimensions must be > 1");
            create_decomp_info(&shape.x, &fi, &rank, &ret[0].x);
            xstart = ret[0]; xend = ret[1]; xsize = ret[2];
            ystart = ret[3]; yend = ret[4]; ysize = ret[5];
            zstart = ret[6]; zend = ret[7]; zsize = ret[8];
        }
        bool compatible_with_complex(const DecompInfo& other) const {
            // sanity check: for the R2C and C2R transforms, the decomposition is x-complete
            // and along the y and z dimensions the real and cmpl decompositions mus tbe the same
            bool comp = xsize.y ==other.xsize.y  and xsize.z ==other.xsize.z;
            comp     *= xstart.y==other.xstart.y and xstart.z==other.xstart.z;
            comp     *= xend.y  ==other.xend.y   and xend.z  ==other.xend.z;
            return comp;
        }
        bool operator==(const DecompInfo& other) const { return fi == other.fi; }
        bool operator!=(const DecompInfo& other) const { return fi != other.fi; }
    };

    std::ostream& operator<<(std::ostream& os, const DecompInfo& di){
        os << "decomp info #" << di.fi << " @ rank " << di.rank << "\n";
        os << "x-pencil from " << di.xstart << "\tto " << di.xend << ",\tsize " << di.xsize << "\n";
        os << "y-pencil from " << di.ystart << "\tto " << di.yend << ",\tsize " << di.ysize << "\n";
        os << "z-pencil from " << di.zstart << "\tto " << di.zend << ",\tsize " << di.zsize << "\n";
        return os;
    }
    template <typename F>
    class DecompArrayInterface {
    public:
        typedef F               RT;
        typedef std::complex<F> CT;
        virtual RT* real_ptr() = 0;
        virtual CT* cmpl_ptr() = 0;
    };
    template <class N>
    class DefaultDecompArrayBase : public N {
    protected:
        using typename N::RT;
        using typename N::CT;
        CT* ptr;
        size_t alloc_bytes;
    public:
        DefaultDecompArrayBase() = delete;
        DefaultDecompArrayBase(DecompInfo cd){
            size_t alloc_len = std::max(std::max(cd.xsize.prod(), cd.ysize.prod()), cd.zsize.prod());
            alloc_bytes = sizeof(CT) * alloc_len;
            ptr = new CT[alloc_len]{};
            int lock = mlock(ptr, alloc_bytes);
            if(lock) fprintf(stderr, "memory region cannot be pinned\n");
        }
        ~DefaultDecompArrayBase(){ delete[] ptr; }
        RT* real_ptr(){ return reinterpret_cast<RT*>(ptr); }
        CT* cmpl_ptr(){ return reinterpret_cast<CT*>(ptr); }
    };
    template <typename F, template <class Interface> class Base = DefaultDecompArrayBase>
    class DecompArray : public Base<DecompArrayInterface<F>> {
        enum DecompType { XD = 1, YD, ZD };
        enum AccessType { RA = 1, CA };
        DecompType dectype;
        AccessType acctype;
    public:
        const DecompInfo realdec, cmpldec;
        DecompArray() = delete;
        DecompArray(const DecompArray& from) : DecompArray(from.realdec, from.cmpldec){}
        DecompArray(int3 shape) : DecompArray(DecompInfo(shape), DecompInfo(shape.real_to_hermitian())){}
        bool is_real() const { return acctype == RA; }
        bool is_cmpl() const { return acctype == CA; }
        bool is_x()    const { return dectype == XD; }
        bool is_y()    const { return dectype == YD; }
        bool is_z()    const { return dectype == ZD; }
        DecompArray& as_real(){ acctype = RA; return *this; }
        DecompArray& as_cmpl(){ acctype = CA; return *this; }
        DecompArray& as_x(){    dectype = XD; return *this; }
        DecompArray& as_y(){    dectype = YD; return *this; }
        DecompArray& as_z(){    dectype = ZD; return *this; }
        void operator>>(DecompArray<F, Base>& to){
            if(not (realdec==to.realdec and cmpldec==to.cmpldec and acctype==to.acctype))
                ERROR("arrays cannot be globally transposed");
            //std::cerr << "realdec " << to.realdec << "\ncmpldec " << to.cmpldec << std::endl;
            //std::cerr << "from " << *this << " to " << to << std::endl;
            if(is_real() and to.is_real()){
                ERROR("not implemented, as only complex arrays require global transposition for fft");
            }
                //global_transposition(this->real_ptr(), dectype, to.real_ptr(), to.dectype, realdec.get_index());
            else if(is_cmpl() and to.is_cmpl())
                global_transposition(this->real_ptr(), dectype, to.real_ptr(), to.dectype, cmpldec.get_index());
            else ERROR("elem type must be the same for global transposition");
        }
    protected:
        DecompArray(DecompInfo _realdec, DecompInfo _cmpldec) :
            Base<DecompArrayInterface<F>>(_cmpldec),
            dectype(XD), acctype(RA),
            realdec(_realdec), cmpldec(_cmpldec){
            if(not realdec.compatible_with_complex(cmpldec)) ERROR("real and cmpl decompositions cannot work together"); }
        void save(std::string fn){
            save_array(this->real_ptr(), (acctype==RA ? 1 : 2) * sizeof(F),
                       acctype==RA ? realdec.get_index() : cmpldec.get_index(),
                       dectype, fn.c_str(), fn.size());
        }
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
    };

    template <typename F, template <class> class B>
    std::ostream& operator<<(std::ostream& os, const DecompArray<F, B>& da){
        char at = '_', dt = '_';
        if(da.is_real()) at = 'R';
        else if(da.is_cmpl()) at = 'C';
        if(da.is_x()) dt = 'X';
        else if(da.is_y()) dt = 'Y';
        else if(da.is_z()) dt = 'Z';
        os << "<array (" << at << ") elem type and (" << dt << ") decomp type>";
        return os;
    }

    template <typename F, class Derived>
    class DistributedFFTBase {
    public:
        template <template <class> class B>
        void r2c(DecompArray<F, B>& a, DecompArray<F, B>& b){
            if(not (a.is_real() and b.is_cmpl() and a.is_x() and b.is_z()))
                ERROR("sanity check failed in r2c()");
            // STEP 1
            static_cast<Derived*>(this)->forward(a, b.as_x()); // x-fft, real -> cmpl
            // STEP 2
            //std::cerr << "from " << b << " to " << a << std::endl;
            b >> a.as_cmpl().as_y(); // x -> y
            // STEP 3
            static_cast<Derived*>(this)->forward(a, b.as_y()); // y-fft, cmpl -> cmpl
            // STEP 4
            //std::cerr << "from " << b << " to " << a << std::endl;
            b >> a.as_z(); // y -> z
            // STEP 5
            static_cast<Derived*>(this)->forward(a, b.as_z()); // z-fft, cmpl -> cmpl
            a.as_x().as_real();
        }
        template <template <class> class B>
        void c2r(DecompArray<F, B>& a, DecompArray<F, B>& b){
            if(not (a.is_cmpl() and b.is_real() and a.is_z() and b.is_x()))
                ERROR("sanity check failed in c2r()");
            // STEP 1
            static_cast<Derived*>(this)->backward(a, b.as_z().as_cmpl()); // z-ifft, cmpl -> cmpl
            // STEP 2
            //std::cerr << "from " << b << " to " << a << std::endl;
            b >> a.as_y(); // z -> y
            // STEP 3
            static_cast<Derived*>(this)->backward(a, b.as_y()); // y-ifft, cmpl -> cmpl
            // STEP 4
            //std::cerr << "from " << b << " to " << a << std::endl;
            b >> a.as_x(); // y -> x
            // STEP 5
            static_cast<Derived*>(this)->backward(a, b.as_x().as_real()); // x-ifft, cmpl -> real
            a.as_z();
        }
    };

    template <typename F>
    class DistributedFFT : public DistributedFFTBase<F, DistributedFFT<F>> {
    public:
        void forward(DecompArray<F>& in, DecompArray<F>& out){
            std::cerr << "dummy forward from " << in << " to " << out << std::endl; }
        void backward(DecompArray<F>& in, DecompArray<F>& out){
            std::cerr << "dummy backward from " << in << " to " << out << std::endl; }
    };
}

namespace Decomp {
    using DecompImpl::DecompArray;
    using DecompImpl::DistributedFFT;
    using DecompImpl::Bookkeeping;
}
