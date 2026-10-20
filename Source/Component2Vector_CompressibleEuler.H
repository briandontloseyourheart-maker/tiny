#ifndef COMPONENT2VECTOR_COMPRESSIBLE_EULER_H_
#define COMPONENT2VECTOR_COMPRESSIBLE_EULER_H_

#include <AMReX_Array.H>//包含数组类型
#include "Variables_CompressibleEuler.H"//引用变量布局

// 将各个标量分量装入数组
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Component2Vector(
    amrex::GpuArray<amrex::Real, NCONS>& Vector,

    amrex::Real const& Density,

    AMREX_D_DECL(
    amrex::Real const& Momentum_x,
    amrex::Real const& Momentum_y,
    amrex::Real const& Momentum_z),

    amrex::Real const& Energy)//读取
{
    Vector[URHO] = Density;
    Vector[UMX]  = Momentum_x;

#if (AMREX_SPACEDIM > 1)
    Vector[UMY] = Momentum_y;
#endif

#if (AMREX_SPACEDIM > 2)
    Vector[UMZ] = Momentum_z;
#endif

    Vector[UENE] = Energy;//修改
}

#endif 