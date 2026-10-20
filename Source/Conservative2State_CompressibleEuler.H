#ifndef CONSERVATIVE2STATE_COMPRESSIBLE_EULER_H_
#define CONSERVATIVE2STATE_COMPRESSIBLE_EULER_H_

#include "Vector2Component_CompressibleEuler.H"

// 从守恒数组取出各个标量状态
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Conservative2State(
    // 输出：各个守恒分量
    amrex::Real& rho,

    AMREX_D_DECL(
    amrex::Real& rho_u,
    amrex::Real& rho_v,
    amrex::Real& rho_w),

    amrex::Real& rho_E,

    // 输入：守恒变量数组
    amrex::GpuArray<amrex::Real, NCONS> const& Conservative)
{
    Vector2Component(
        rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
        rho_E,
        Conservative
    );
}

#endif