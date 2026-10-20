#ifndef STATE2CONSERVATIVE_COMPRESSIBLE_EULER_H_
#define STATE2CONSERVATIVE_COMPRESSIBLE_EULER_H_

#include "Component2Vector_CompressibleEuler.H"

// 将已计算好的守恒状态装入数组
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void State2Conservative(
    // 输出：守恒变量数组
    amrex::GpuArray<amrex::Real, NCONS>& Conservative,

    // 输入：各个守恒分量
    amrex::Real const& rho,

    AMREX_D_DECL(
    amrex::Real const& rho_u,
    amrex::Real const& rho_v,
    amrex::Real const& rho_w),//保证函数参数数量与调用时传入的数量一致

    amrex::Real const& rho_E)
{
    Component2Vector(
        Conservative,
        rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
        rho_E
    );
}

#endif