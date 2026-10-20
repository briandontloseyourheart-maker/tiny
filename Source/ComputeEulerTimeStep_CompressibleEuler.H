#ifndef COMPUTE_EULER_TIMESTEP_COMPRESSIBLE_EULER_H_
#define COMPUTE_EULER_TIMESTEP_COMPRESSIBLE_EULER_H_

#include <cmath>
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include "Conservative2Primitive_CompressibleEuler.H"
// 根据有效单元状态估计声学CFL时间步
inline void ComputeEulerTimeStep(
    amrex::Real& dt_out,

    amrex::MultiFab const& U,
    amrex::Geometry const& geom,
    amrex::Real const CFL,
    amrex::Real const gamma)
    //检查输入条件
{
    if (!(CFL > 0.0 && CFL <= 1.0) ||
        !(gamma > 1.0)) {
        amrex::Abort("Invalid CFL or gamma");
    }

    if (!(U.min(URHO) > 0.0)) {
        amrex::Abort("Nonpositive density in Euler state");
    }

    amrex::MultiFab Primitive(
        U.boxArray(), U.DistributionMap(), NPRIM, 0
    );

    amrex::MultiFab Rate(
        U.boxArray(), U.DistributionMap(), 1, 0
    );

    auto const dX = geom.CellSizeArray();

    // 先恢复有效单元的原始状态
    for (amrex::MFIter mfi(U); mfi.isValid(); ++mfi) {
        auto const ua = U.const_array(mfi);
        auto const va = Primitive.array(mfi);
        amrex::Box const bx = mfi.validbox();

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::GpuArray<amrex::Real, NCONS> U_cell{};
                amrex::GpuArray<amrex::Real, NPRIM> V_cell{};
                amrex::Real e;

                for (int n = 0; n < NCONS; ++n) {
                    U_cell[n] = ua(i, j, k, n);
                }

                Conservative2Primitive(
                    V_cell, e, U_cell, gamma
                );

                for (int n = 0; n < NPRIM; ++n) {
                    va(i, j, k, n) = V_cell[n];
                }
            }
        );
    }

    amrex::Gpu::streamSynchronize();

    if (!(Primitive.min(VP) > 0.0)) {
        amrex::Abort("Nonpositive pressure in Euler state");
    }

    // 计算各方向传播率之和
    for (amrex::MFIter mfi(Primitive); mfi.isValid(); ++mfi) {
        auto const va = Primitive.const_array(mfi);
        auto const ra = Rate.array(mfi);
        amrex::Box const bx = mfi.validbox();

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real rho = va(i, j, k, VRHO);
                amrex::Real p   = va(i, j, k, VP);
                amrex::Real c   = std::sqrt(gamma * p / rho);

                amrex::Real rate = 0.0;

                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    amrex::Real velocity = va(i, j, k, 1 + d);

                    amrex::Real abs_velocity =
                        (velocity >= 0.0)
                        ? velocity
                        : -velocity;

                    rate += (abs_velocity + c) / dX[d];
                }

                ra(i, j, k) = rate;
            }
        );
    }

    amrex::Gpu::streamSynchronize();

    // norminf默认进行全局归约，不只是当前MPI进程
    amrex::Real rate_max = Rate.norminf(0);//取最大值
//最大传播率是有限的正数
    if (!(rate_max > 0.0) || !std::isfinite(rate_max)) {
        amrex::Abort("Invalid acoustic propagation rate");
    }

    dt_out = CFL / rate_max;
}
#endif