#ifndef CONSERVATIVE2PRIMITIVE_COMPRESSIBLE_EULER_H_
#define CONSERVATIVE2PRIMITIVE_COMPRESSIBLE_EULER_H_

#include "Conservative2State_CompressibleEuler.H"
#include "State2Primitive_CompressibleEuler.H"

// 守恒变量数组 → 原始变量数组
// 要求 rho > 0、rho_e > 0、gamma > 1
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Conservative2Primitive(
    // 输出
    amrex::GpuArray<amrex::Real, NPRIM>& Primitive,
    amrex::Real& e,

    // 输入
    amrex::GpuArray<amrex::Real, NCONS> const& Conservative,
    amrex::Real const gamma)
{
    // 1. 准备接收守恒分量
    amrex::Real rho;
    amrex::Real rho_E;

    AMREX_D_TERM(
    amrex::Real rho_u;,
    amrex::Real rho_v;,
    amrex::Real rho_w;);

    // 2. 从守恒数组取出标量
    Conservative2State(
        rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
        rho_E,
        Conservative
    );

    // 3. 动量除以密度，恢复速度
    AMREX_D_TERM(
    amrex::Real u = rho_u / rho;,
    amrex::Real v = rho_v / rho;,
    amrex::Real w = rho_w / rho;);

    // 4. 总能量密度减去动能密度
    amrex::Real rho_e =
        rho_E
        - 0.5 * rho * (
            AMREX_D_TERM(u*u, + v*v, + w*w)
        );

    // 5. 比内能与理想气体压力
    e = rho_e / rho;
    amrex::Real p = (gamma - 1.0) * rho_e;

    // 6. 将密度、速度、压力装入原始数组
    State2Primitive(
        Primitive,
        rho,
        AMREX_D_DECL(u, v, w),
        p
    );
}

#endif