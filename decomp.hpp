#pragma once
#include <mpi.h>
#include <sys/mman.h>
#include <string>
#include <complex>
#include <random>
#include <functional>
#include "interop.h"

void assert(bool cond, const char* file, const int line, const char* msg = NULL){
    if(not cond){
        if(msg)
            std::fprintf(stderr, "ERR @ %s:%d > %s\n", file, line, msg);
        else
            std::fprintf(stderr, "ERR @ %s:%d\n", file, line);
        std::exit(EXIT_FAILURE);
    }
}
#define ASSERTMSG(stuff, msg) assert(stuff, __FILE__, __LINE__, msg)
#define ASSERT(stuff) assert(stuff, __FILE__, __LINE__)
#define MASTER if(DecompImpl::rank() == 0)

namespace DecompGlobals {

    using Decomp::RT;
    using Decomp::CT;

    bool context_started = false;
    int size = 0, rank = 0, seed = 0, rank_in_node = 0, size_of_node = 0;
}

namespace DecompImpl {

    using Decomp::RT;
    using Decomp::CT;

    int size(){         return DecompGlobals::size; }
    int rank(){         return DecompGlobals::rank; }
    int seed(){         return DecompGlobals::seed; }
    int rank_in_node(){ return DecompGlobals::rank_in_node; }
    int size_of_node(){ return DecompGlobals::size_of_node; }

    struct int3 {
        int x,y,z;
        int3() = default;
        int3(int _x, int _y, int _z){ x=_x; y=_y; z=_z; };
        typedef int* IPTR;
        operator IPTR() const { return IPTR(this); }
        int prod() const { return x * y * z; }
        int3 halfcomplex() const { return {x/2+1, y, z}; }
        void size_t3(size_t shape[3]){ shape[0] = x; shape[1] = y; shape[2] = z; }
    } __attribute__((packed));
    std::ostream& operator<<(std::ostream& os, const int3& i3){
        os << "(" << i3.x << "," << i3.y << "," << i3.z << ")";
        return os;
    }
    bool operator==(const int3& a, const int3& b){ return a.x==b.x and a.y==b.y and a.z==b.z; }

    void start_decomp_context(int3 gshape){
        if(DecompGlobals::context_started){
            std::cerr << "decomp context already started" << std::endl;
            return;
        }
        MPI::Init();
        DecompGlobals::size = MPI::COMM_WORLD.Get_size();
        DecompGlobals::rank = MPI::COMM_WORLD.Get_rank();

        std::random_device rd;
        srand(rd());
        srand(rd() * rand());
        srand(rd() * rand());
        DecompGlobals::seed = rand();
        srand(seed());
        std::cout << "rank (" << rank() << ") says seed (" << seed() << ")" << std::endl;

        char pname[MPI_MAX_PROCESSOR_NAME]{};
        int pnamelen;
        MPI::Get_processor_name(pname, pnamelen);

        char* othernames = NULL;
        if(rank() == 0) othernames = new char[size() * MPI_MAX_PROCESSOR_NAME]{};

        MPI::COMM_WORLD.Gather(pname, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, othernames, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, 0);

        int* id_by_rank{};
        int* clique_size{};
        if(rank() == 0){
            id_by_rank = new int[size()];
            int _pairid[size()];
            int* pairid_ptr[size()];
            int* pairid = _pairid;
            for(int id = 0; id < size(); id++) { id_by_rank[id] = -1; }
            for(int id = 0; id < size(); id++){
                if(id_by_rank[id] != -1) continue; // pair already found
                *pairid = 0;
                id_by_rank[id] = *pairid;
                pairid_ptr[id] = pairid;
                for(int other = 0; other < size(); other++){
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
            clique_size = new int[size()];
            for(int id = 0; id < size(); id++) { clique_size[id] = *pairid_ptr[id]+1; }
        }
        int id_within_node, node_size;
        MPI::COMM_WORLD.Scatter(id_by_rank, 1, MPI_INT, &id_within_node, 1, MPI_INT, 0);
        MPI::COMM_WORLD.Scatter(clique_size, 1, MPI_INT, &node_size, 1, MPI_INT, 0);
        delete[] id_by_rank;
        delete[] clique_size;
        delete[] othernames;
        DecompGlobals::size_of_node = node_size;
        DecompGlobals::rank_in_node = id_within_node;
        std::cerr << "rank (" << rank << ") has id ("
                  << id_within_node << ") within clique of size ("
                  << node_size << ")" << std::endl;

        // initialize 2DECOMP with processor grid given below, {0,0} is to autotune
        int proc_grid[2] = {0, 0};
        initialize(gshape, proc_grid);

        // checking if library float size equals to program's
        int float_size;
        get_float_size(&float_size);
        if(float_size != sizeof(RT)){
            fprintf(stderr, "sizeof(F) = %d , mytype_bytes = %d\n", sizeof(RT), float_size);
            ASSERTMSG(false, "main() and 2DECOMP float sizes differ");
        }
        DecompGlobals::context_started = true;
    }
    void end_decomp_context(){
        if(not DecompGlobals::context_started){
            std::cerr << "decomp context has not been started" << std::endl;
            return;
        }
        finalize();
        MPI::Finalize();
        DecompGlobals::context_started = false;
    }
    bool check_if_homogeneous(char* buffer, int len){
        int* lens{};
        if(rank() == 0) lens = new int[size()];
        MPI::COMM_WORLD.Gather(&len, 1, MPI_INT, lens, 1, MPI_INT, 0);
        if(rank() == 0)
            for(int id = 0; id < size(); id++)
                ASSERTMSG(lens[id] == len, "len parameter in check_if_homogeneous is non-homogeneous");

        char* bigstring{};
        if(rank() == 0) bigstring = new char[size() * len];
        MPI::COMM_WORLD.Gather(buffer, len, MPI_BYTE, bigstring, len, MPI_BYTE, 0);
        if(rank() == 0){
            bool check = true;
            for(int id = 0; id < size(); id++){
                if(0 != strncmp(buffer, bigstring + id * len, len)){
                    check = false;
                    break;
                }
            }
            for(int id = 0; id < size(); id++){ lens[id] = check; }
        }
        int ret;
        MPI::COMM_WORLD.Scatter(lens, 1, MPI_INT, &ret, 1, MPI_INT, 0);
        delete[] lens, bigstring;
        return ret;
    }

    class _DecompInfo {
        int fi;
    public:
        int get_index() const { return fi; }
        int3 xstart, xend, xsize, ystart, yend, ysize, zstart, zend, zsize;
        _DecompInfo() = delete;
        _DecompInfo(const _DecompInfo& other) = default;
        _DecompInfo(int3 shape){
            ASSERTMSG(DecompGlobals::context_started, "decomp context has not been started");
            int3 ret[3 * 3] = {};
            int _rank = rank();
            create_decomp_info(&shape.x, &fi, &_rank, &ret[0].x);
            xstart = ret[0]; xend = ret[1]; xsize = ret[2];
            ystart = ret[3]; yend = ret[4]; ysize = ret[5];
            zstart = ret[6]; zend = ret[7]; zsize = ret[8];
        }
        bool operator==(const _DecompInfo& other) const { return fi == other.fi; }
        bool operator!=(const _DecompInfo& other) const { return fi != other.fi; }
    };

    std::ostream& operator<<(std::ostream& os, const _DecompInfo& di){
        os << "decomp info #" << di.get_index() << " @ rank " << rank() << "\n";
        os << "x-pencil from " << di.xstart << "\tto " << di.xend << ",\tsize " << di.xsize << "\n";
        os << "y-pencil from " << di.ystart << "\tto " << di.yend << ",\tsize " << di.ysize << "\n";
        os << "z-pencil from " << di.zstart << "\tto " << di.zend << ",\tsize " << di.zsize << "\n";
        return os;
    }

    class DecompInfo {
    public:
        const _DecompInfo realdec, cmpldec;
        DecompInfo() = delete;
        DecompInfo(const DecompInfo&) = default;
        DecompInfo(int3 realshape) : realdec(realshape), cmpldec(realshape.halfcomplex()){
            ASSERTMSG(realshape.x > 1 and realshape.y > 1 and realshape.z > 1, "dimensions must be > 1");
            // sanity check: for the R2C and C2R transforms, the decomposition is x-complete
            // and along the y and z dimensions the real and cmpl decompositions mus tbe the same
            bool comp = realdec.xsize.y  == cmpldec.xsize.y  and realdec.xsize.z  == cmpldec.xsize.z;
            comp     *= realdec.xstart.y == cmpldec.xstart.y and realdec.xstart.z == cmpldec.xstart.z;
            comp     *= realdec.xend.y   == cmpldec.xend.y   and realdec.xend.z   == cmpldec.xend.z;
            ASSERTMSG(comp, "real and complex decompositions are not compatible with each other");
        }
        bool operator==(const DecompInfo& other) const {
            return realdec.get_index()==other.realdec.get_index() and cmpldec.get_index()==other.cmpldec.get_index(); }
    };
    std::ostream& operator<<(std::ostream& os, const DecompInfo& di){
        os << "real: " << di.realdec << "cmpl: " << di.cmpldec;
        return os;
    }


    class MemoryMan {
    public:
        class ContextMan {
            CT* const ptr;
        public:
            ContextMan() = delete;
            ContextMan(const ContextMan&) = delete;
            ContextMan(CT* _ptr) : ptr(_ptr){}
            ContextMan(ContextMan&& cm) : ptr(std::move(cm.ptr)){}
            template <typename T> operator T*(){ return (T*)ptr; }
        };
        CT* ptr;
    public:
        size_t alloc_bytes;
        MemoryMan(DecompInfo di){
            size_t alloc_len = std::max(std::max(di.cmpldec.xsize.prod(), di.cmpldec.ysize.prod()), di.cmpldec.zsize.prod());
            ptr = new CT[alloc_len]{};
            alloc_bytes = sizeof(CT) * alloc_len;
            int lock = mlock(ptr, alloc_bytes);
            if(lock) fprintf(stderr, "memory region cannot be pinned\n");
        }
        ~MemoryMan(){ delete[] ptr; }
        ContextMan operator()() const { return ContextMan(ptr); }
    };

    template <class MM>
    class DecompArray {
        enum DecompType { XD = 1, YD, ZD };
        enum AccessType { RA = 1, CA };
        DecompType dectype;
        AccessType acctype;
    public:
        const DecompInfo decinfo;
        MM mm;
        DecompArray() = delete;
        DecompArray(const DecompArray& from) :
            dectype(from.dectype), acctype(from.acctype), decinfo(from.decinfo), mm(decinfo){}
        DecompArray(int3 realshape) :
            dectype(XD), acctype(RA), decinfo(realshape), mm(decinfo){}
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
        void operator>>(DecompArray<MM>& to){
            ASSERTMSG(decinfo==to.decinfo and acctype==to.acctype, "arrays cannot be globally transposed");
            if(is_real() and to.is_real()){
                ASSERTMSG(false, "not implemented, as only complex arrays require global transposition for fft");
                //global_transposition(this->real_ptr(), dectype, to.real_ptr(), to.dectype, realdec.get_index());
            } else if(is_cmpl() and to.is_cmpl()){
                auto mfrom = mm();
                auto mto = to.mm();
                global_transposition(mfrom, dectype, mto, to.dectype, decinfo.cmpldec.get_index());
            } else ASSERTMSG(false, "elem type must be the same for global transposition");
        }

        void save(std::string fn){
            auto man = mm();
            save_array(man, (acctype==RA ? 1 : 2) * sizeof(RT),
                       acctype==RA ? decinfo.realdec.get_index() : decinfo.cmpldec.get_index(),
                       dectype, fn.c_str(), fn.size());
        }
    };

    template <typename T, class MM>
    void over(DecompArray<MM>& arr, std::function<void (const int&, const int&, const int&, T&)> closure){
        const _DecompInfo* di = NULL;
        if(arr.is_real() and sizeof(T)==sizeof(RT))
            di = &arr.decinfo.realdec;
        else if(arr.is_cmpl() and sizeof(T)==sizeof(CT))
            di = &arr.decinfo.cmpldec;
        ASSERTMSG(di != NULL, "closure argument type and array type dont match");
        int ix, iy, iz, ixg, iyg, izg, ixst, iyst, izst, xsz, ysz, zsz;
        if(arr.is_x()){
            ixst = di->xstart.x; iyst = di->xstart.y; izst = di->xstart.z;
            xsz = di->xsize.x; ysz = di->xsize.y; zsz = di->xsize.z;
        } else if(arr.is_y()){
            ixst = di->ystart.x; iyst = di->ystart.y; izst = di->ystart.z;
            xsz = di->ysize.x; ysz = di->ysize.y; zsz = di->ysize.z;
        } else if(arr.is_z()){
            ixst = di->zstart.x; iyst = di->zstart.y; izst = di->zstart.z;
            xsz = di->zsize.x; ysz = di->zsize.y; zsz = di->zsize.z;
        } else ASSERTMSG(false, "cannot happen");
        auto man = arr.mm();
        T* ptr = man;
        for(iz = 0; iz < zsz; iz++){
            izg = izst + iz;
            for(iy = 0; iy < ysz; iy++){
                iyg = iyst + iy;
                for(ix = 0; ix < xsz; ix++){
                    ixg = ixst + ix;
                    closure(ixg, iyg, izg, *ptr);
                    ptr++; }}}
    }

    template <class MM>
    std::ostream& operator<<(std::ostream& os, const DecompArray<MM>& da){
        char at = '_', dt = '_';
        if(da.is_real()) at = 'R';
        else if(da.is_cmpl()) at = 'C';
        if(da.is_x()) dt = 'X';
        else if(da.is_y()) dt = 'Y';
        else if(da.is_z()) dt = 'Z';
        os << "<array (" << at << ") elem type and (" << dt << ") decomp type of (" << std::round(da.mm.alloc_bytes / 1024 / 1024) << ") MBs>";
        return os;
    }

    template <class Derived>
    class DistributedFFTBase {
    public:
        template <class MM>
        void r2c(DecompArray<MM>& a, DecompArray<MM>& b){
            ASSERTMSG(a.is_real() and b.is_cmpl() and a.is_x() and b.is_z(), "sanity check failed in r2c()");
            // STEP 1
            tm1.start();
            static_cast<Derived*>(this)->forward(a, b.as_x()); // x-fft, real -> cmpl
            tm1.stop(false);
            // STEP 2
            tm3.start();
            b >> a.as_cmpl().as_y(); // x -> y
            // STEP 3
            tm3.start();
            static_cast<Derived*>(this)->forward(a, b.as_y()); // y-fft, cmpl -> cmpl
            tm3.stop(false);
            // STEP 4
            tm3.start();
            b >> a.as_z(); // y -> z
            tm3.stop(true);
            // STEP 5
            tm1.start();
            static_cast<Derived*>(this)->forward(a, b.as_z()); // z-fft, cmpl -> cmpl
            tm1.stop(true);
            a.as_x().as_real();
        }
        template <class MM>
        void c2r(DecompArray<MM>& a, DecompArray<MM>& b){
            ASSERTMSG(a.is_cmpl() and b.is_real() and a.is_z() and b.is_x(), "sanity check failed in c2r()");
            // STEP 1
            tm2.start();
            static_cast<Derived*>(this)->backward(a, b.as_z().as_cmpl()); // z-ifft, cmpl -> cmpl
            tm2.stop(false);
            // STEP 2
            tm4.start();
            b >> a.as_y(); // z -> y
            tm4.stop(false);
            // STEP 3
            tm2.start();
            static_cast<Derived*>(this)->backward(a, b.as_y()); // y-ifft, cmpl -> cmpl
            tm2.stop(false);
            // STEP 4
            tm4.start();
            b >> a.as_x(); // y -> x
            tm4.stop(true);
            // STEP 5
            tm2.start();
            static_cast<Derived*>(this)->backward(a, b.as_x().as_real()); // x-ifft, cmpl -> real
            tm2.stop(true);
            a.as_z();
        }
    };

    class DistributedFFT : public DistributedFFTBase<DistributedFFT> {
    public:
        template <class MM> void forward(DecompArray<MM>& in, DecompArray<MM>& out){
            std::cerr << "dummy forward from " << in << " to " << out << std::endl; }
        template <class MM> void backward(DecompArray<MM>& in, DecompArray<MM>& out){
            std::cerr << "dummy backward from " << in << " to " << out << std::endl; }
    };
}

namespace Decomp {
    using DecompImpl::int3;
    //using DecompImpl::start_decomp_context;
    //using DecompImpl::end_decomp_context;
    //using DecompImpl::size;
    //using DecompImpl::rank;
    //using DecompImpl::seed;
    //using DecompImpl::rank_in_node;
    //using DecompImpl::size_of_node;
    using DecompArray = DecompImpl::DecompArray<DecompImpl::MemoryMan>;
    using DecompImpl::over;
    using DecompImpl::DistributedFFT;
}
