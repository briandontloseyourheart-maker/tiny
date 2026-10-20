#ifndef VECTOR2COMPONENT_COMPRESSIBLE_EULER_H_
#define VECTOR2COMPONENT_COMPRESSIBLE_EULER_H_

#include <AMReX_Array.H>
#include "Variables_CompressibleEuler.H"

// 从数组取出各个标量分量

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void Vector2Component(
    amrex::Real& Density,

    AMREX_D_DECL(
    amrex::Real& Momentum_x,
    amrex::Real& Momentum_y,
    amrex::Real& Momentum_z),

    amrex::Real& Energy,//总能量密度

    amrex::GpuArray<amrex::Real, NCONS> const& Vector)
{
    Density    = Vector[URHO];
    Momentum_x = Vector[UMX];

#if (AMREX_SPACEDIM > 1)
    Momentum_y = Vector[UMY];
#endif

#if (AMREX_SPACEDIM > 2)
    Momentum_z = Vector[UMZ];
#endif

    Energy = Vector[UENE];
}

#endif
//Density    = rho
//Momentum_x = rho*u
//Momentum_y = rho*v
//Momentum_z = rho*w
//Energy     = rho*E