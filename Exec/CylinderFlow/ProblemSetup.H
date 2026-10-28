#ifndef ProblemSetup_H_
#define ProblemSetup_H_

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Solid_Fraction (amrex::Real& Alpha_Solid,
        AMREX_D_DECL(amrex::Real const& x,
                     amrex::Real const& y,
                     amrex::Real const& z),
                     amrex::Real const& time)
{
    AMREX_D_TERM(
    amrex::Real xr = 0.0;,
    amrex::Real yr = 0.0;,
    amrex::Real zr = 0.0);
    amrex::Real rr = 0.5;
    amrex::Real psi = rr - std::sqrt(AMREX_D_TERM((x-xr)*(x-xr), + (y-yr)*(y-yr), + (z-zr)*(z-zr)));
    Alpha_Solid = 0.5*(1.0 + std::tanh(psi/0.01));
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Solid_Velocity (amrex::Array<amrex::Real,AMREX_SPACEDIM>& Velocity_Solid,
        AMREX_D_DECL(amrex::Real const& x,
                     amrex::Real const& y,
                     amrex::Real const& z),
                     amrex::Real const& time)
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
        Velocity_Solid[idim] = 0.0;
    }
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
                        amrex::Real const& z))
{
    AMREX_D_TERM(
    u = 0.0001;,
    v = 0.0001;,
    w = 0.001);
    P = 0.0;
}
#endif
