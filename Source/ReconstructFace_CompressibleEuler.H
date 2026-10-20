#ifndef RECONSTRUCTFACE_COMPRESSIBLE_EULER_H_
#define RECONSTRUCTFACE_COMPRESSIBLE_EULER_H_

#include <AMReX_Array4.H>
#include <AMReX_Array.H>
#include <AMReX.H>

#include "Variables_CompressibleEuler.H"
#include "Limiter_Minmod.H"
#include "Conservative2Primitive_CompressibleEuler.H"

// 在一个网格面上构造左右原始状态
// 面索引约定：
// x方向面(i,j,k)位于单元(i-1,j,k)与(i,j,k)之间
// y方向面(i,j,k)位于单元(i,j-1,k)与(i,j,k)之间
// z方向面(i,j,k)位于单元(i,j,k-1)与(i,j,k)之间
//
// Primitive 必须包含已正确填充的两层ghost。
// 要求：0 <= ID_Dimension < AMREX_SPACEDIM。
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void ReconstructFace(
    // 输出：界面左右原始状态
    amrex::GpuArray<amrex::Real, NPRIM>& PrimitiveL,
    amrex::GpuArray<amrex::Real, NPRIM>& PrimitiveR,

    // 输入：单元中心原始变量场
    amrex::Array4<amrex::Real const> const& Primitive,

    // 输入：面索引、网格间距、方向
    int const i,
    int const j,
    int const k,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dX,
    int const ID_Dimension)//当前重构方向，0 为 x、1 为 y、2 为 z
{
    // 当前方向的一个单元偏移
    int di = (ID_Dimension == 0) ? 1 : 0;
    int dj = (ID_Dimension == 1) ? 1 : 0;
    int dk = (ID_Dimension == 2) ? 1 : 0;

    // 界面左侧单元
    int il = i - di;
    int jl = j - dj;
    int kl = k - dk;

    // 界面右侧单元
    int ir = i;
    int jr = j;
    int kr = k;

    amrex::Real dx = dX[ID_Dimension];

    // 对所有原始变量分别重构
    for (int n = 0; n < NPRIM; ++n) {
        amrex::Real fL = Primitive(il, jl, kl, n);
        amrex::Real fR = Primitive(ir, jr, kr, n);

        // 左侧单元的两个候选斜率
        amrex::Real slopeL_minus =
            (fL - Primitive(il-di, jl-dj, kl-dk, n)) / dx;

        amrex::Real slopeL_plus =
            (Primitive(il+di, jl+dj, kl+dk, n) - fL) / dx;

        amrex::Real slopeL;

        Limiter_Minmod(
            slopeL,
            slopeL_minus,
            slopeL_plus
        );

        // 右侧单元的两个候选斜率
        amrex::Real slopeR_minus =
            (fR - Primitive(ir-di, jr-dj, kr-dk, n)) / dx;

        amrex::Real slopeR_plus =
            (Primitive(ir+di, jr+dj, kr+dk, n) - fR) / dx;

        amrex::Real slopeR;

        Limiter_Minmod(
            slopeR,
            slopeR_minus,
            slopeR_plus
        );

        // 左单元向右重构，右单元向左重构
        PrimitiveL[n] = fL + 0.5 * dx * slopeL;
        PrimitiveR[n] = fR - 0.5 * dx * slopeR;
    }
}

// 按需重构版本：输入守恒场，额外传入gamma。
// 只把当前面模板上的四个单元转换为局部原始状态，不创建整场Primitive。
// 保留上面的原始场接口，已有调用不受影响。
// 直接读取守恒场，临时转换四个模板单元再重构
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void ReconstructFace(
    // 输出
    amrex::GpuArray<amrex::Real, NPRIM>& PrimitiveL,
    amrex::GpuArray<amrex::Real, NPRIM>& PrimitiveR,

    // 输入
    amrex::Array4<amrex::Real const> const& Conservative,
    int const i,
    int const j,
    int const k,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dX,
    int const ID_Dimension,
    amrex::Real const gamma)
{
    int const di = (ID_Dimension == 0) ? 1 : 0;
    int const dj = (ID_Dimension == 1) ? 1 : 0;
    int const dk = (ID_Dimension == 2) ? 1 : 0;

    amrex::Real const dx = dX[ID_Dimension];

    // 只保存当前面需要的四个单元的原始状态
    amrex::GpuArray<
        amrex::GpuArray<amrex::Real, NPRIM>, 4
    > V{};

    for (int s = 0; s < 4; ++s) {
        int const offset = s - 2;

        amrex::GpuArray<amrex::Real, NCONS> U_cell{};

        for (int n = 0; n < NCONS; ++n) {
            U_cell[n] = Conservative(
                i + offset * di,
                j + offset * dj,
                k + offset * dk,
                n
            );
        }

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            U_cell[URHO] > 0.0,
            "Nonpositive density in reconstruction stencil"
        );

        amrex::Real e;

        Conservative2Primitive(
            V[s], e, U_cell, gamma
        );

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            V[s][VP] > 0.0,
            "Nonpositive pressure in reconstruction stencil"
        );
    }

    // 对每个原始变量分量分别限制和重构
    for (int n = 0; n < NPRIM; ++n) {
        amrex::Real slopeL;
        amrex::Real slopeR;

        Limiter_Minmod(
            slopeL,
            (V[1][n] - V[0][n]) / dx,
            (V[2][n] - V[1][n]) / dx
        );

        Limiter_Minmod(
            slopeR,
            (V[2][n] - V[1][n]) / dx,
            (V[3][n] - V[2][n]) / dx
        );

        PrimitiveL[n] =
            V[1][n] + 0.5 * dx * slopeL;

        PrimitiveR[n] =
            V[2][n] - 0.5 * dx * slopeR;
    }
}

#endif
