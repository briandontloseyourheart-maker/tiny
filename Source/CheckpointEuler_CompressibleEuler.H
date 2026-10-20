#ifndef CHECKPOINT_EULER_COMPRESSIBLE_EULER_H_
#define CHECKPOINT_EULER_COMPRESSIBLE_EULER_H_

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#include "Variables_CompressibleEuler.H"

// 写入新Euler格式；输出为文件
inline void WriteEulerCheckpoint(
    amrex::MultiFab const& U,
    amrex::Geometry const& geom,
    int const step,
    amrex::Real const time,
    amrex::Real const gamma,
    amrex::Real const gas_constant,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& lo_bc,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& hi_bc)
{
    std::string const dirname =
        amrex::Concatenate("./checkpoint_euler/chk", step, 5);

    // 与原检查点采用相同的目录创建机制
    // 同名目录存在时，AMReX会将旧目录重命名保存
    amrex::PreBuildDirectorHierarchy(
        dirname, "Level_", 1, true
    );

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream header(dirname + "/Header");

        if (!header) {
            amrex::Abort("Cannot create Euler checkpoint Header");
        }

        header.precision(17);

        header << "EulerCheckpoint_v1\n";
        header << AMREX_SPACEDIM << " " << NCONS << "\n";
        header << step << " " << time << "\n";
        header << gamma << " " << gas_constant << "\n";

        header << geom.Domain() << "\n";

        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            header
                << geom.ProbLo(d) << " "
                << geom.ProbHi(d) << " "
                << (geom.isPeriodic(d) ? 1 : 0) << " "
                << lo_bc[d] << " "
                << hi_bc[d] << "\n";
        }

        U.boxArray().writeOn(header);
        header << "\n";

        header.close();

        if (!header) {
            amrex::Abort("Failed writing Euler checkpoint Header");
        }
    }

    amrex::ParallelDescriptor::Barrier();

    // 只保存有效单元，不保存ghost
    amrex::MultiFab U_disk(
        U.boxArray(), U.DistributionMap(), NCONS, 0
    );

    amrex::MultiFab::Copy(
        U_disk, U, 0, 0, NCONS, 0
    );

    amrex::VisMF::Write(
        U_disk, dirname + "/Level_0/U"
    );

    amrex::ParallelDescriptor::Barrier();

    amrex::Print()
        << "Wrote Euler checkpoint: " << dirname << "\n";
}

// 读取新Euler格式
// 输出在前，输入在后
inline void ReadEulerCheckpoint(
    amrex::MultiFab& U,
    amrex::BoxArray& ba,
    amrex::DistributionMapping& dm,
    amrex::Real& time,
    int& saved_step,

    int const restart,
    amrex::Geometry const& geom,
    amrex::Real const gamma,
    amrex::Real const gas_constant,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& lo_bc,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& hi_bc)
{
    std::string const dirname =
        amrex::Concatenate("./checkpoint_euler/chk", restart, 5);

    amrex::Vector<char> buffer;

    amrex::ParallelDescriptor::ReadAndBcastFile(
        dirname + "/Header", buffer
    );

    std::string const text(buffer.dataPtr());
    std::istringstream header(text);

    std::string magic;
    std::getline(header, magic);

    if (magic != "EulerCheckpoint_v1") {
        amrex::Abort("Not a supported Euler checkpoint");
    }

    int saved_dim;
    int saved_ncons;
    amrex::Real saved_gamma;
    amrex::Real saved_R;

    header >> saved_dim >> saved_ncons;
    header >> saved_step >> time;
    header >> saved_gamma >> saved_R;

    if (!header ||
        saved_dim != AMREX_SPACEDIM ||
        saved_ncons != NCONS ||
        saved_step != restart) {
        amrex::Abort("Euler checkpoint metadata mismatch");
    }

    auto same_real = [](amrex::Real a, amrex::Real b) {
        return std::abs(a-b) <= 1.0e-12 * (1.0 + std::abs(b));
    };

    if (!same_real(saved_gamma, gamma) ||
        !same_real(saved_R, gas_constant)) {
        amrex::Abort("Restart must use the same gamma and gas_constant");
    }

    amrex::Box saved_domain;
    header >> saved_domain;

    if (!(saved_domain == geom.Domain())) {
        amrex::Abort("Restart grid domain differs from checkpoint");
    }

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        amrex::Real saved_lo;
        amrex::Real saved_hi;
        int saved_periodic;
        int saved_lo_bc;
        int saved_hi_bc;

        header
            >> saved_lo >> saved_hi
            >> saved_periodic
            >> saved_lo_bc >> saved_hi_bc;

        if (!header ||
            !same_real(saved_lo, geom.ProbLo(d)) ||
            !same_real(saved_hi, geom.ProbHi(d)) ||
            saved_periodic != (geom.isPeriodic(d) ? 1 : 0) ||
            saved_lo_bc != lo_bc[d] ||
            saved_hi_bc != hi_bc[d]) {
            amrex::Abort("Restart geometry or boundary conditions differ");
        }
    }

    header >> std::ws;
    ba.readFrom(header);

    if (!header) {
        amrex::Abort("Invalid checkpoint BoxArray");
    }

    // 使用当前MPI进程数重新分配网格块
    dm.define(ba, amrex::ParallelDescriptor::NProcs());

    amrex::MultiFab U_disk(ba, dm, NCONS, 0);

    amrex::VisMF::Read(
        U_disk, dirname + "/Level_0/U"
    );

    if (U_disk.nComp() != NCONS ||
        !(U_disk.boxArray() == ba)) {
        amrex::Abort("Checkpoint state layout mismatch");
    }

    U.define(ba, dm, NCONS, 2);
    U.setVal(0.0);

    amrex::MultiFab::Copy(
        U, U_disk, 0, 0, NCONS, 0
    );

    // ghost将在下一次AdvanceEuler中重新填充
    amrex::Print()
        << "Read Euler checkpoint: " << dirname
        << ", time = " << time << "\n";
}

#endif