#ifndef OUTPUT_EULER_COMPRESSIBLE_EULER_H_
#define OUTPUT_EULER_COMPRESSIBLE_EULER_H_

#include <cmath>

#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "Conservative2Primitive_CompressibleEuler.H"

// 输出是plotfile，因此此函数的参数均为输入
inline void WriteEulerPlot(
    amrex::MultiFab const& U,
    amrex::Geometry const& geom,
    int const step,
    amrex::Real const time,
    amrex::Real const gamma,
    amrex::Real const gas_constant)
{
    if (!(gamma > 1.0) || !(gas_constant > 0.0)) {
        amrex::Abort("Invalid thermodynamic parameters");
    }

    if (!(U.min(URHO) > 0.0)) {
        amrex::Abort("Cannot output nonpositive density");
    }

    // 守恒量 + 原始变量（不重复密度）+ e、T、Mach
    int const e_index = NCONS + NPRIM - 1;
    int const t_index = e_index + 1;
    int const mach_index = e_index + 2;
    int const noutput = e_index + 3;

    amrex::MultiFab Output(
        U.boxArray(), U.DistributionMap(), noutput, 0
    );

    amrex::Vector<std::string> names = vnamesC;

    for (int n = 1; n < NPRIM; ++n) {
        names.push_back(vnamesP[n]);
    }

    names.push_back("e");
    names.push_back("T");
    names.push_back("Mach");

    for (amrex::MFIter mfi(U); mfi.isValid(); ++mfi) {
        auto const ua = U.const_array(mfi);
        auto const oa = Output.array(mfi);
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
                    oa(i, j, k, n) = U_cell[n];
                }

                Conservative2Primitive(
                    V_cell, e, U_cell, gamma
                );

                for (int n = 1; n < NPRIM; ++n) {
                    oa(i, j, k, NCONS + n - 1) = V_cell[n];
                }

                amrex::Real rho = V_cell[VRHO];
                amrex::Real p = V_cell[VP];
                amrex::Real speed_squared = 0.0;

                for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                    speed_squared +=
                        V_cell[1+d] * V_cell[1+d];
                }

                oa(i, j, k, e_index) = e;
                oa(i, j, k, t_index) =
                    p / (rho * gas_constant);

                // 负压力将在写文件前报错，不截断压力继续计算
                oa(i, j, k, mach_index) =
                    (p > 0.0)
                    ? std::sqrt(speed_squared / (gamma*p/rho))
                    : 0.0;
            }
        );
    }

    amrex::Gpu::streamSynchronize();

    int const p_index = NCONS + VP - 1;

    if (!(Output.min(p_index) > 0.0)) {
        amrex::Abort("Cannot output nonpositive pressure");
    }

    std::string const filename =
        amrex::Concatenate("./plot_euler/plt", step, 5);

    amrex::WriteSingleLevelPlotfile(
        filename, Output, names, geom, time, step
    );
}

#endif