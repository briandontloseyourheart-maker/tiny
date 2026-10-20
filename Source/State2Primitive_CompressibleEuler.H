#ifndef STATE2PRIMITIVE_COMPRESSIBLE_EULER_H_
#define STATE2PRIMITIVE_COMPRESSIBLE_EULER_H_

#include "Component2Vector_CompressibleEuler.H"

// 将已计算好的原始状态装入数组
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void State2Primitive(
    // 输出：原始变量数组
    amrex::GpuArray<amrex::Real, NPRIM>& Primitive,

    // 输入：密度、速度、压力
    amrex::Real const& rho,

    AMREX_D_DECL(
    amrex::Real const& u,
    amrex::Real const& v,
    amrex::Real const& w),

    amrex::Real const& p)
{
    Component2Vector(
        Primitive,
        rho,
        AMREX_D_DECL(u, v, w),
        p
    );
}

#endif