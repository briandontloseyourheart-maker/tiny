#ifndef Variables_H_
#define Variables_H_

#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>
#include <string>

inline constexpr int NSTATE = 2 + AMREX_SPACEDIM;
inline amrex::Vector<std::string> vnamesC = {"rho", AMREX_D_DECL("rho_u", "rho_v", "rho_w"), "rho_E"};
inline amrex::Vector<std::string> vnamesD = {"rho", AMREX_D_DECL(    "u",     "v",     "w"),     "P"};
enum StateIndex{
    INDEX_Mass      = 0,
    AMREX_D_DECL(
    INDEX_MomentumX = 1,
    INDEX_MomentumY = 2,
    INDEX_MomentumZ = 3),
    INDEX_Energy    = 1+AMREX_SPACEDIM
};


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Component2Vector(amrex::Array<amrex::Real,NSTATE>& Vector,
                      amrex::Real const& Mass,
         AMREX_D_DECL(amrex::Real const& Momentum_x,
                      amrex::Real const& Momentum_y,
                      amrex::Real const& Momentum_z),
                      amrex::Real const& Energy)
{
    Vector[INDEX_Mass]      = Mass;
    Vector[INDEX_MomentumX] = Momentum_x;
#if (AMREX_SPACEDIM > 1)
    Vector[INDEX_MomentumY] = Momentum_y;
#endif
#if (AMREX_SPACEDIM > 2)
    Vector[INDEX_MomentumZ] = Momentum_z;
#endif
    Vector[INDEX_Energy]    = Energy;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Vector2Component(amrex::Real& Mass,
         AMREX_D_DECL(amrex::Real& Momentum_x,
                      amrex::Real& Momentum_y,
                      amrex::Real& Momentum_z),
                      amrex::Real& Energy,
                      amrex::Array<amrex::Real,NSTATE> const& Vector)
{
    Mass       = Vector[INDEX_Mass];
    Momentum_x = Vector[INDEX_MomentumX];
#if (AMREX_SPACEDIM > 1)
    Momentum_y = Vector[INDEX_MomentumY];
#endif
#if (AMREX_SPACEDIM > 2)
    Momentum_z = Vector[INDEX_MomentumZ];
#endif
    Energy     = Vector[INDEX_Energy];
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Conservative2State(amrex::Real& rho,
           AMREX_D_DECL(amrex::Real& rho_u,
                        amrex::Real& rho_v,
                        amrex::Real& rho_w),
                        amrex::Real& rho_E,
                        amrex::Array<amrex::Real,NSTATE> const& Conservative)
{
    Vector2Component(rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
                     rho_E,
                     Conservative);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void State2Conservative(amrex::Array<amrex::Real,NSTATE>& Conservative,
                        amrex::Real const& rho,
           AMREX_D_DECL(amrex::Real const& rho_u,
                        amrex::Real const& rho_v,
                        amrex::Real const& rho_w),
                        amrex::Real const& rho_E)
{
    Component2Vector(Conservative,
                     rho,
        AMREX_D_DECL(rho_u, rho_v, rho_w),
                     rho_E);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Primitive2State(amrex::Real& rho,
        AMREX_D_DECL(amrex::Real& u,
                     amrex::Real& v,
                     amrex::Real& w),
                     amrex::Real& p,
                     amrex::Array<amrex::Real,NSTATE> const& Primitive)
{
    Vector2Component(rho,
        AMREX_D_DECL(u, v, w),
                     p,
                     Primitive);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void State2Primitive(amrex::Array<amrex::Real,NSTATE>& Primitive,
                     amrex::Real const& rho,
        AMREX_D_DECL(amrex::Real const& u,
                     amrex::Real const& v,
                     amrex::Real const& w),
                     amrex::Real const& p)
{
    Component2Vector(Primitive,
                     rho,
        AMREX_D_DECL(u, v, w),
                     p);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Conservative2Primitive(amrex::Array<amrex::Real,NSTATE>& Primitive,
                            amrex::Array<amrex::Real,NSTATE> const& Conservative,
                            amrex::Real const& gamma)
{
    amrex::Real rho;
    AMREX_D_TERM( amrex::Real rho_u;, 
                  amrex::Real rho_v;, 
                  amrex::Real rho_w );
    amrex::Real rho_E;
    Conservative2State(rho,
          AMREX_D_DECL(rho_u, rho_v, rho_w),
                       rho_E,
                       Conservative);
    
    AMREX_D_TERM( amrex::Real u = rho_u/rho;,
                  amrex::Real v = rho_v/rho;,
                  amrex::Real w = rho_w/rho );
    amrex::Real rho_e = rho_E - (0.5)*rho*( AMREX_D_TERM( u*u,
                                                         +v*v,
                                                         +w*w ) );
    
    amrex::Real p = (gamma - 1.0)*rho_e;
    
    State2Primitive(Primitive,
                    rho,
       AMREX_D_DECL(u, v, w),
                    p);
    
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Primitive2Conservative(amrex::Array<amrex::Real,NSTATE>& Conservative,
                            amrex::Array<amrex::Real,NSTATE> const& Primitive,
                            amrex::Real const& gamma)
{
    amrex::Real rho;
    AMREX_D_TERM(amrex::Real u;, 
                 amrex::Real v;, 
                 amrex::Real w );
    amrex::Real p;
    Primitive2State(rho,
       AMREX_D_DECL(u, v, w),
                    p,
                    Primitive);
    
    AMREX_D_TERM( amrex::Real rho_u = rho*u;,
                  amrex::Real rho_v = rho*v;,
                  amrex::Real rho_w = rho*w );
    
    amrex::Real rho_e = 1.0/(gamma - 1.0)*p;
    
    amrex::Real rho_E = rho_e + (0.5)*rho*( AMREX_D_TERM( u*u,
                                                         +v*v,
                                                         +w*w ) );
    
    State2Conservative(Conservative,
                       rho,
          AMREX_D_DECL(rho_u, rho_v, rho_w),
                       rho_E);
    
}
#endif