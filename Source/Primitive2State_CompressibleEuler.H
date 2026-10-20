#ifndef PRIMITIVE2STATE_COMPRESSIBLE_EULER_H_
#define PRIMITIVE2STATE_COMPRESSIBLE_EULER_H_

#include "Vector2Component_CompressibleEuler.H"

// 从原始变量数组取出各个标量
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Primitive2State(
    // 输出：密度、速度、压力
    amrex::Real& rho,

    AMREX_D_DECL(
    amrex::Real& u,
    amrex::Real& v,
    amrex::Real& w),

    amrex::Real& p,

    // 输入：原始变量数组
    amrex::GpuArray<amrex::Real, NPRIM> const& Primitive)
{
    Vector2Component(
        rho,
        AMREX_D_DECL(u, v, w),
        p,
        Primitive
    );
}

#endif