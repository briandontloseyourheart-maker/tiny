#ifndef EULER_FLUX_AND_UPDATE_COMPRESSIBLE_EULER_H_
#define EULER_FLUX_AND_UPDATE_COMPRESSIBLE_EULER_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include "Conservative2Primitive_CompressibleEuler.H"
#include "Primitive2Conservative_CompressibleEuler.H"
#include "ReconstructFace_CompressibleEuler.H"
#include "Riemann_LLF_CompressibleEuler.H"

// 单层、Cartesian网格上的Euler通量计算与前向Euler更新
// U_old的两层ghost必须在调用前正确填充。
// U_new、U_old必须使用相同的单元BoxArray和DistributionMapping。
// Flux必须按各方向的面BoxArray预先分配。
// U_old和U_new必须是不同的状态数组。
// dt必须已经满足所采用格式的稳定性限制。
inline void EulerFluxAndUpdate(
    // 输出
    amrex::MultiFab& U_new,
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Flux,

    // 输入
    amrex::MultiFab const& U_old,
    amrex::Geometry const& geom,
    amrex::Real const dt,
    amrex::Real const gamma)
    //检查
{
    if (U_old.nComp() != NCONS ||//检查分量数
        U_new.nComp() != NCONS ||
        U_old.nGrow() < 2) {//检查是否有两层ghost
        amrex::Abort(
            "EulerFluxAndUpdate: incorrect components or ghost layers"
        );
    }

    if (dt <= 0.0 || gamma <= 1.0) {
        amrex::Abort(
            "EulerFluxAndUpdate requires dt > 0 and gamma > 1"
        );
    }

    // 检查待转换区域的密度，包括两层ghost
    if (!(U_old.min(URHO, 2) > 0.0)) {
        amrex::Abort("Nonpositive density in U_old or ghost cells");
    }

    auto const dX = geom.CellSizeArray();
// A. 先遍历网格块，再遍历方向，逐面计算通量
for (amrex::MFIter mfi(U_old); mfi.isValid(); ++mfi) {
    auto const uold_array = U_old.const_array(mfi);
    amrex::Box const cell_bx = mfi.validbox();

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        auto const flux_array = Flux[d].array(mfi);

        // 单元范围 → 当前方向的面范围
        amrex::Box bx = cell_bx;
        bx.surroundingNodes(d);

        int const direction = d;

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::GpuArray<amrex::Real, NPRIM> VL{};
                amrex::GpuArray<amrex::Real, NPRIM> VR{};

                amrex::GpuArray<amrex::Real, NCONS> UL{};
                amrex::GpuArray<amrex::Real, NCONS> UR{};

                amrex::GpuArray<amrex::Real, NCONS> F_face{};
                amrex::GpuArray<
                    amrex::Real, AMREX_SPACEDIM
                > velocity_face{};

                amrex::Real cmax_face;

                // 直接读取守恒场，按需转换并重构
                ReconstructFace(
                    VL,
                    VR,
                    uold_array,
                    i, j, k,
                    dX,
                    direction,
                    gamma
                );

                // 左右原始状态 → 左右守恒状态
                Primitive2Conservative(UL, VL, gamma);
                Primitive2Conservative(UR, VR, gamma);

                // 左右守恒状态 → 一个界面数值通量
                Riemann_LLF(
                    F_face,
                    velocity_face,
                    cmax_face,
                    UL,
                    UR,
                    gamma,
                    direction
                );

                // 保存当前面的各分量通量
                for (int n = 0; n < NCONS; ++n) {
                    flux_array(i, j, k, n) = F_face[n];
                }
            }
        );
    }
}

// B. 全部面计算完成后，统一重复面的通量
amrex::Gpu::streamSynchronize();

for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    Flux[d].OverrideSync(geom.periodicity());
}
    

    // C. 根据面通量差更新内部守恒量，公式与原版本相同。
    for (amrex::MFIter mfi(U_old); mfi.isValid(); ++mfi) {
        auto const uold_array = U_old.const_array(mfi);
        auto const unew_array = U_new.array(mfi);

        auto const fx = Flux[0].const_array(mfi);

#if (AMREX_SPACEDIM > 1)
        auto const fy = Flux[1].const_array(mfi);
#endif

#if (AMREX_SPACEDIM > 2)
        auto const fz = Flux[2].const_array(mfi);
#endif

        amrex::Box const bx = mfi.validbox();

        amrex::ParallelFor(
            bx,
            NCONS,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                amrex::Real divergence =
                    (fx(i+1, j, k, n) - fx(i, j, k, n))
                    / dX[0];

#if (AMREX_SPACEDIM > 1)
                divergence +=
                    (fy(i, j+1, k, n) - fy(i, j, k, n))
                    / dX[1];
#endif

#if (AMREX_SPACEDIM > 2)
                divergence +=
                    (fz(i, j, k+1, n) - fz(i, j, k, n))
                    / dX[2];
#endif

                unew_array(i, j, k, n) =
                    uold_array(i, j, k, n)
                    - dt * divergence;
            }
        );
    }

    amrex::Gpu::streamSynchronize();
}

#endif
