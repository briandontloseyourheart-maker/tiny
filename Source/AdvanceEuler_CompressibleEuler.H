#ifndef ADVANCE_EULER_COMPRESSIBLE_EULER_H_
#define ADVANCE_EULER_COMPRESSIBLE_EULER_H_

#include "ComputeEulerTimeStep_CompressibleEuler.H"
#include "BoundaryEuler_CompressibleEuler.H"
#include "EulerFluxAndUpdate_CompressibleEuler.H"

inline void AdvanceEuler(
    amrex::MultiFab& U_new,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,

    amrex::MultiFab& U_old,
    amrex::Geometry const& geom,
    amrex::Real const dt,
    amrex::Real const gamma,

    amrex::GpuArray<int, AMREX_SPACEDIM> const& lo_bc,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& hi_bc)
{
    // 填充本次推进所需的旧守恒状态ghost
    BoundaryEuler(U_old, geom, lo_bc, hi_bc);

    // 重构、LLF、面同步、守恒更新
    EulerFluxAndUpdate(
        U_new,
        Flux,
        U_old,
        geom,
        dt,
        gamma
    );
}

#endif