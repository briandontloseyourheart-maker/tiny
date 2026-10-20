#ifndef PRIMITIVE2CONSERVATIVE_COMPRESSIBLE_EULER_H_
#define PRIMITIVE2CONSERVATIVE_COMPRESSIBLE_EULER_H_

#include "Primitive2State_CompressibleEuler.H"
#include "State2Conservative_CompressibleEuler.H"

// 原始变量数组 → 守恒变量数组
// 要求 rho > 0、p > 0、gamma > 1
// 输出在前，输入在后
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Primitive2Conservative(
    // 输出
    amrex::GpuArray<amrex::Real, NCONS>& Conservative,//输出数组，把结果写入

    // 输入
    amrex::GpuArray<amrex::Real, NPRIM> const& Primitive,
    amrex::Real const gamma)
{
    // 1. 准备接收原始分量
    amrex::Real rho;
    amrex::Real p;

    AMREX_D_TERM(
    amrex::Real u;,
    amrex::Real v;,
    amrex::Real w;);

    // 2. 从原始数组取出密度、速度、压力
    Primitive2State(
        rho,
        AMREX_D_DECL(u, v, w),
        p,
        Primitive
    );

    // 3. 计算各方向动量密度
    AMREX_D_TERM(
    amrex::Real rho_u = rho * u;,
    amrex::Real rho_v = rho * v;,
    amrex::Real rho_w = rho * w;);

    // 4. 理想气体内能密度
    amrex::Real rho_e = p / (gamma - 1.0);

    // 5. 总能量密度 = 内能密度 + 动能密度
    amrex::Real rho_E =
        rho_e
        + 0.5 * rho * (
            AMREX_D_TERM(u*u, + v*v, + w*w)
        );

    // 6. 将守恒分量装入数组
    State2Conservative(
        Conservative,
        rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
        rho_E
    );
}

#endif