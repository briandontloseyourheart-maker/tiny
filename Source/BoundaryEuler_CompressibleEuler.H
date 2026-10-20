#ifndef BOUNDARY_EULER_COMPRESSIBLE_EULER_H_
#define BOUNDARY_EULER_COMPRESSIBLE_EULER_H_

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>

#include "Variables_CompressibleEuler.H"

// 当前支持的Euler边界类型
enum EulerBoundaryType {
    EULER_PERIODIC  = 0,
    EULER_SLIP_WALL = 1
};

// 填充内部、周期及固定滑移壁面的两层ghost
inline void BoundaryEuler(
    amrex::MultiFab& U,

    amrex::Geometry const& geom,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& lo_bc,
    amrex::GpuArray<int, AMREX_SPACEDIM> const& hi_bc)
{
    if (U.nComp() != NCONS || U.nGrow() < 2) {
        amrex::Abort("BoundaryEuler requires NCONS and two ghost layers");
    }

    amrex::GpuArray<int, AMREX_SPACEDIM> periodic{};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_lo{};
    amrex::GpuArray<int, AMREX_SPACEDIM> domain_hi{};

    // 检查边界设置与Geometry是否一致
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        periodic[d] = geom.isPeriodic(d) ? 1 : 0;
        domain_lo[d] = geom.Domain().smallEnd(d);
        domain_hi[d] = geom.Domain().bigEnd(d);

        if (periodic[d]) {
            if (lo_bc[d] != EULER_PERIODIC ||
                hi_bc[d] != EULER_PERIODIC) {
                amrex::Abort("Periodic Geometry requires periodic Euler BC");
            }
        }
        else {
            // 当前非周期方向只支持两端固定滑移壁
            if (lo_bc[d] != EULER_SLIP_WALL ||
                hi_bc[d] != EULER_SLIP_WALL) {
                amrex::Abort("Only periodic and slip-wall BC are implemented");
            }

            if (geom.Domain().length(d) < 2) {
                amrex::Abort("Slip-wall direction needs at least two cells");
            }
        }
    }

    // 先填内部通信与周期ghost
    U.FillBoundary(geom.periodicity());

    // 再填非周期物理边界ghost
    // 不使用tiling，避免重复写入同一个Fab的扩展区域
    for (amrex::MFIter mfi(U); mfi.isValid(); ++mfi) {
        auto const ua = U.array(mfi);

        amrex::Box bx = mfi.validbox();
        bx.grow(2);

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::GpuArray<int, 3> position{i, j, k};
                amrex::GpuArray<int, 3> source{i, j, k};

                amrex::GpuArray<int, AMREX_SPACEDIM> sign{};

                bool outside_physical = false;

                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    sign[d] = 1;

                    // 周期方向已经由FillBoundary处理
                    if (periodic[d]) {
                        continue;
                    }

                    if (position[d] < domain_lo[d]) {
                        // 低侧壁面镜像
                        source[d] =
                            2 * domain_lo[d] - 1 - position[d];

                        sign[d] = -1;
                        outside_physical = true;
                    }
                    else if (position[d] > domain_hi[d]) {
                        // 高侧壁面镜像
                        source[d] =
                            2 * domain_hi[d] + 1 - position[d];

                        sign[d] = -1;
                        outside_physical = true;
                    }
                }

                // 不修改内部单元或纯周期ghost
                if (!outside_physical) {
                    return;
                }

                for (int n = 0; n < NCONS; ++n) {
                    amrex::Real value =
                        ua(source[0], source[1], source[2], n);

                    // 仅镜像方向的动量反号
                    if (n >= UMX &&
                        n < UMX + AMREX_SPACEDIM) {
                        value *= sign[n - UMX];
                    }

                    ua(i, j, k, n) = value;
                }
            }
        );
    }

    amrex::Gpu::streamSynchronize();
}

#endif