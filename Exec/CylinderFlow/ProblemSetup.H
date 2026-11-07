#ifndef ProblemSetup_H_
#define ProblemSetup_H_

#include "Variables.H"
#include "mystruct.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Solid_Fraction (amrex::Real& Alpha_Solid,
        AMREX_D_DECL(amrex::Real const& x,
                     amrex::Real const& y,
                     amrex::Real const& z),
                     amrex::Real const& time,
                     ImmersedBoundaryData const& immersed_boundary_data)
{
    AMREX_D_TERM(
    amrex::Real xr = immersed_boundary_data.x(time);,
    amrex::Real yr = immersed_boundary_data.y(time);,
    amrex::Real zr = immersed_boundary_data.z(time));
    amrex::Real rr = immersed_boundary_data.radius;
    amrex::Real epsilon = immersed_boundary_data.epsilon;
    amrex::Real psi = rr - std::sqrt(AMREX_D_TERM((x-xr)*(x-xr), + (y-yr)*(y-yr), + (z-zr)*(z-zr)));
    Alpha_Solid = 0.5*(1.0 + std::tanh(psi/epsilon));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Solid_Velocity (amrex::Array<amrex::Real,AMREX_SPACEDIM>& Velocity_Solid,
        AMREX_D_DECL(amrex::Real const& x,
                     amrex::Real const& y,
                     amrex::Real const& z),
                     amrex::Real const& time,
                     ImmersedBoundaryData const& immersed_boundary_data)
{
    amrex::Real dt = 1.0e-6*std::max(time, 1.0);
    Velocity_Solid = immersed_boundary_data.Velocity(time, dt);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Drag_Coefficient (amrex::Real& Cd,
                       amrex::Real const& Alpha_Solid,
                       amrex::Real const& dt)
{
    Cd = Alpha_Solid/dt;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Initial_Condition (AMREX_D_DECL(
                        amrex::Real& u,
                        amrex::Real& v,
                        amrex::Real& w),
                        amrex::Real& P,
                        AMREX_D_DECL(
                        amrex::Real const& x,
                        amrex::Real const& y,
                        amrex::Real const& z),
                        InitialConditionData const& initial_condition_data)
{
    AMREX_D_TERM(
    u = initial_condition_data.u(AMREX_D_DECL(x, y, z));,
    v = initial_condition_data.v(AMREX_D_DECL(x, y, z));,
    w = initial_condition_data.w(AMREX_D_DECL(x, y, z)));
    P = initial_condition_data.P(AMREX_D_DECL(x, y, z));
}
#endif
