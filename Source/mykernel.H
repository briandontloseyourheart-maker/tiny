#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include "ProblemSetup.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Conservative2Flux(amrex::Array<amrex::Real,NSTATE>& Flux,
                       amrex::Array<amrex::Real,NSTATE> const& Conservative,
                       amrex::Real const& gamma,
                       int ID_Dimension)
{
    amrex::Array<amrex::Real,NSTATE> Primitive;
    Conservative2Primitive(Primitive,
                           Conservative,
                           gamma);
    
    amrex::Real rho;
    AMREX_D_TERM( amrex::Real u;,
                  amrex::Real v;,
                  amrex::Real w );
    amrex::Real p;
    Primitive2State(rho,
       AMREX_D_DECL(u, v, w),
                    p,
                    Primitive);

    AMREX_D_TERM( amrex::Real rho_u;,
                  amrex::Real rho_v;,
                  amrex::Real rho_w );
    amrex::Real rho_E;
    Conservative2State(rho,
          AMREX_D_DECL(rho_u, rho_v, rho_w),
                       rho_E,
                       Conservative);
    
    amrex::Real V;
    //x
    if (ID_Dimension == 0){
        V = u;
    }
#if (AMREX_SPACEDIM > 1)
    //y
    else if (ID_Dimension == 1){
        V = v;
    }
#endif
#if (AMREX_SPACEDIM > 2)    
    //z
    else if (ID_Dimension == 2){
        V = w;
    }
#endif

    Component2Vector(Flux,
                     rho*V,
        AMREX_D_DECL(rho_u*V, rho_v*V, rho_w*V),
                     (rho_E + p)*V);
    Flux[INDEX_MomentumX+ID_Dimension] += p;
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Riemann_LLF(amrex::Array<amrex::Real,NSTATE>& Flux_Riemann,
                 amrex::Real& cmax,
                 amrex::Array<amrex::Real,NSTATE> const& ConservativeL,
                 amrex::Array<amrex::Real,NSTATE> const& ConservativeR,
                 amrex::Real const& gamma,
                 int ID_Dimension)
{
    amrex::Array<amrex::Real,NSTATE> PrimitiveL;
    Conservative2Primitive(PrimitiveL,
                           ConservativeL,
                           gamma);
    
    amrex::Array<amrex::Real,NSTATE> PrimitiveR;
    Conservative2Primitive(PrimitiveR,
                           ConservativeR,
                           gamma);
    
    amrex::Array<amrex::Real,NSTATE> FluxL;
    Conservative2Flux(FluxL,
                      ConservativeL,
                      gamma,
                      ID_Dimension);
    
    amrex::Array<amrex::Real,NSTATE> FluxR;
    Conservative2Flux(FluxR,
                      ConservativeR,
                      gamma,
                      ID_Dimension);
    
    amrex::Real rhoL;
    AMREX_D_TERM( amrex::Real uL;,
                  amrex::Real vL;,
                  amrex::Real wL );
    amrex::Real pL;
    Primitive2State(rhoL,
       AMREX_D_DECL(uL, vL, wL),
                    pL,
                    PrimitiveL);

    amrex::Real rhoR;
    AMREX_D_TERM( amrex::Real uR;,
                  amrex::Real vR;,
                  amrex::Real wR );
    amrex::Real pR;
    Primitive2State(rhoR,
       AMREX_D_DECL(uR, vR, wR),
                    pR,
                    PrimitiveR);
    
    amrex::Real cL = std::sqrt(gamma*pL/rhoL);
    amrex::Real cR = std::sqrt(gamma*pR/rhoR);
    
    amrex::Real VL;
    amrex::Real VR;
	//x
    if (ID_Dimension == 0){
        VL = uL;
        VR = uR;
    }
#if (AMREX_SPACEDIM > 1)
    //y
    else if (ID_Dimension == 1){
        VL = vL;
        VR = vR;
    }
#endif
#if (AMREX_SPACEDIM > 2)      
    //z
    else if (ID_Dimension == 2){
        VL = wL;
        VR = wR;
    }
#endif
    
    amrex::Real Lambda = std::max( std::abs(VL) + cL, std::abs(VR) + cR );
    for (int i_State = 0; i_State < NSTATE; ++i_State){
        Flux_Riemann[i_State] = ( FluxL[i_State] + FluxR[i_State] - Lambda*(ConservativeR[i_State] - ConservativeL[i_State]) )/2.0;
    }

    cmax = std::max(cL + std::sqrt(AMREX_D_TERM(uL*uL, + vL*vL, + wL*wL)), cR + std::sqrt(AMREX_D_TERM(uR*uR, + vR*vR, + wR*wR)));

}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Limiter_Minmod(amrex::Real& Output,
                    amrex::Real const& Input_a,
                    amrex::Real const& Input_b)
{
    Output = std::min( std::max(Input_a,0.0),std::max(Input_b,0.0) )
            +std::max( std::min(Input_a,0.0),std::min(Input_b,0.0) );
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Reconstruction2Face_MUSCL(amrex::Real& f_Face,
                               amrex::Real const& f_m,
                               amrex::Real const& f_c,
                               amrex::Real const& f_p,
                               int ID_Face)
{
    amrex::Real df_m = f_c - f_m;
    amrex::Real df_p = f_p - f_c;
    amrex::Real df;
    Limiter_Minmod(df,
                   df_m, df_p);
    f_Face = f_c + amrex::Real(ID_Face)*0.5*df;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Reconstruction2Face_Conservative_MUSCL(amrex::Array<amrex::Real,NSTATE>& Conservative_Face,
                                            amrex::Array<amrex::Real,NSTATE> const& Conservative_m,
                                            amrex::Array<amrex::Real,NSTATE> const& Conservative_c,
                                            amrex::Array<amrex::Real,NSTATE> const& Conservative_p,
                                            amrex::Real const& gamma,
                                            int ID_Face)
{
    amrex::Array<amrex::Real,NSTATE> Primitive_m;
    Conservative2Primitive(Primitive_m,
                           Conservative_m,
                           gamma);

    amrex::Array<amrex::Real,NSTATE> Primitive_c;
    Conservative2Primitive(Primitive_c,
                           Conservative_c,
                           gamma);

    amrex::Array<amrex::Real,NSTATE> Primitive_p;
    Conservative2Primitive(Primitive_p,
                           Conservative_p,
                           gamma);

    amrex::Array<amrex::Real,NSTATE> Primitive_Face;
    for (int iState = 0; iState < NSTATE; iState++)
    {
        amrex::Real f_m = Primitive_m[iState];
        amrex::Real f_c = Primitive_c[iState];
        amrex::Real f_p = Primitive_p[iState];
        amrex::Real f_Face;
        Reconstruction2Face_MUSCL(f_Face,
                                  f_m, f_c, f_p,
                                  ID_Face);
        Primitive_Face[iState] = f_Face;
    }

    Primitive2Conservative(Conservative_Face,
                           Primitive_Face,
                           gamma);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Reconstruction2FaceLR_Conservative_MUSCL(amrex::Array<amrex::Real,NSTATE>& ConservativeL,
                                              amrex::Array<amrex::Real,NSTATE>& ConservativeR,
                                              amrex::Array<amrex::Real,NSTATE> const& Conservative_mm,
                                              amrex::Array<amrex::Real,NSTATE> const& Conservative_m,
                                              amrex::Array<amrex::Real,NSTATE> const& Conservative_c,
                                              amrex::Array<amrex::Real,NSTATE> const& Conservative_p,
                                              amrex::Real const& gamma)
{
    int ID_Face;

    ID_Face = 1;
    Reconstruction2Face_Conservative_MUSCL(ConservativeL,
                                           Conservative_mm, Conservative_m, Conservative_c,
                                           gamma,
                                           ID_Face);

    ID_Face = -1;
    Reconstruction2Face_Conservative_MUSCL(ConservativeR,
                                           Conservative_m, Conservative_c, Conservative_p,
                                           gamma,
                                           ID_Face);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Reconstruction2FaceLR_Conservative_MUSCL(int i, int j, int k,
                                              amrex::Array<amrex::Real,NSTATE>& ConservativeL,
                                              amrex::Array<amrex::Real,NSTATE>& ConservativeR,
                                              amrex::Array4<amrex::Real const> const& Conservative,
                                              amrex::Real const& gamma,
                                              int ID_Dimension)
{
    amrex::Array<amrex::Real,NSTATE> Conservative_mm;
    amrex::Array<amrex::Real,NSTATE> Conservative_m;
    amrex::Array<amrex::Real,NSTATE> Conservative_c;
    amrex::Array<amrex::Real,NSTATE> Conservative_p;
    for (int iState = 0; iState < NSTATE; iState++)
    {
        Conservative_c[iState] = Conservative(i,j,k,iState);
        if (ID_Dimension == 0){
            Conservative_mm[iState] = Conservative(i-2,j,k,iState);
            Conservative_m [iState] = Conservative(i-1,j,k,iState);
            Conservative_p [iState] = Conservative(i+1,j,k,iState);
        }
        else if (ID_Dimension == 1){
            Conservative_mm[iState] = Conservative(i,j-2,k,iState);
            Conservative_m [iState] = Conservative(i,j-1,k,iState);
            Conservative_p [iState] = Conservative(i,j+1,k,iState);
        }
        else if (ID_Dimension == 2){
            Conservative_mm[iState] = Conservative(i,j,k-2,iState);
            Conservative_m [iState] = Conservative(i,j,k-1,iState);
            Conservative_p [iState] = Conservative(i,j,k+1,iState);
        }
    }
    Reconstruction2FaceLR_Conservative_MUSCL(ConservativeL, ConservativeR,
                                             Conservative_mm, Conservative_m, Conservative_c, Conservative_p,
                                             gamma);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Conservative2FluxHyperbolic(int i, int j, int k,
                                 amrex::Array<amrex::Real,NSTATE>& Flux,
                                 amrex::Real& cmax,
                                 amrex::Array4<amrex::Real const> const& Conservative,
                                 amrex::Real const& gamma,
                                 int ID_Dimension)
{
    amrex::Array<amrex::Real,NSTATE> ConservativeL;
    amrex::Array<amrex::Real,NSTATE> ConservativeR;
    Reconstruction2FaceLR_Conservative_MUSCL(i, j, k,
                                             ConservativeL, ConservativeR,
                                             Conservative,
                                             gamma,
                                             ID_Dimension);

    Riemann_LLF(Flux, cmax,
                ConservativeL, ConservativeR,
                gamma,
                ID_Dimension);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Initial_Condition (int i, int j, int k,
                        amrex::Array4<amrex::Real> const& UArray,
                        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dX,
                        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                        InitialConditionData const& initial_condition_data,
                        amrex::RandomEngine const& random_engine,
                        amrex::Real const& gamma)
{
    AMREX_D_TERM(amrex::Real x = prob_lo[0] + (amrex::Real(i) + 0.5) * dX[0];,
                 amrex::Real y = prob_lo[1] + (amrex::Real(j) + 0.5) * dX[1];,
                 amrex::Real z = prob_lo[2] + (amrex::Real(k) + 0.5) * dX[2]);

    amrex::Real rho, AMREX_D_DECL(u, v, w), P;
    Initial_Condition(rho, AMREX_D_DECL(u, v, w), P,
         AMREX_D_DECL(x, y, z),
                      initial_condition_data);

    amrex::Real rho_Random = amrex::RandomNormal(0.0, initial_condition_data.rho_noise, random_engine);
    amrex::Real u_Random   = amrex::RandomNormal(0.0, initial_condition_data.u_noise  , random_engine);
#if (AMREX_SPACEDIM > 1)
    amrex::Real v_Random   = amrex::RandomNormal(0.0, initial_condition_data.v_noise  , random_engine);
#endif
#if (AMREX_SPACEDIM > 2)
    amrex::Real w_Random   = amrex::RandomNormal(0.0, initial_condition_data.w_noise  , random_engine);
#endif
    amrex::Real P_Random   = amrex::RandomNormal(0.0, initial_condition_data.P_noise  , random_engine);

    rho += rho_Random;
    AMREX_D_TERM(
    u   += u_Random;,
    v   += v_Random;,
    w   += w_Random);
    P   += P_Random;

    amrex::Array<amrex::Real,NSTATE> Primitive;
    State2Primitive(Primitive,
                    rho, AMREX_D_DECL(u, v, w), P);

    amrex::Array<amrex::Real,NSTATE> Conservative;
    Primitive2Conservative(Conservative,
                           Primitive,
                           gamma);

    for (int iState = 0; iState < NSTATE; iState++){
        UArray(i,j,k,iState) = Conservative[iState];
    }
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Ghost_Value (amrex::Real& f_Ghost,
                  amrex::Real const& f,
                  amrex::Real const& BC_Dirichlet,
                  amrex::Real const& BC_Neumann,
                  amrex::Real const& BC_Function,
                  amrex::Real const& CellSize)
{
    f_Ghost = (BC_Function - (BC_Dirichlet/2.0)*f + (BC_Neumann/CellSize)*f)/((BC_Dirichlet/2.0) + (BC_Neumann/CellSize));
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Boundary_Condition (int i, int j, int k,
                         amrex::Array4<amrex::Real> const& fArray,
                         amrex::Array<BoundaryConditionType,AMREX_SPACEDIM> const& BC_lo,
                         amrex::Array<BoundaryConditionType,AMREX_SPACEDIM> const& BC_hi,
                         amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> const& BC_lo_Dirichlet,
                         amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> const& BC_hi_Dirichlet,
                         amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> const& BC_lo_Neumann,
                         amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> const& BC_hi_Neumann,
                         amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> const& BC_lo_Function,
                         amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> const& BC_hi_Function,
                         AMREX_D_DECL(int ilo, int jlo, int klo),
                         AMREX_D_DECL(int ihi, int jhi, int khi),
                         AMREX_D_DECL(amrex::Real const& xlo, amrex::Real const& ylo, amrex::Real const& zlo),
                         AMREX_D_DECL(amrex::Real const& xhi, amrex::Real const& yhi, amrex::Real const& zhi),
                         AMREX_D_DECL(amrex::Real const&  dx, amrex::Real const&  dy, amrex::Real const&  dz),
                         amrex::Real const& time)
{
    int ID_Dimension = 0;
    if (i < ilo){
        if (BC_lo[ID_Dimension] != BoundaryConditionType::Periodic){
            int iInterior = 2*ilo - i - 1;
            int jInterior = j;
            int kInterior = k;
            amrex::Real fInterior = fArray(iInterior, jInterior, kInterior);

            AMREX_D_TERM(
            amrex::Real xBC = xlo                            ;,
            amrex::Real yBC = ylo + (amrex::Real(j) + 0.5)*dy;,
            amrex::Real zBC = zlo + (amrex::Real(k) + 0.5)*dz);

            amrex::Real BC_Dirichlet = BC_lo_Dirichlet[ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Neumann   = BC_lo_Neumann  [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Function  = BC_lo_Function [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real CellSize     = dx;

            amrex::Real f_Ghost;
            Ghost_Value(f_Ghost,
                        fInterior,
                        BC_Dirichlet, BC_Neumann, BC_Function,
                        CellSize);
            fArray(i,j,k) = f_Ghost;
        }
    }
    else if (i > ihi){
        if (BC_hi[ID_Dimension] != BoundaryConditionType::Periodic){
            int iInterior = 2*ihi - i + 1;
            int jInterior = j;
            int kInterior = k;
            amrex::Real fInterior = fArray(iInterior, jInterior, kInterior);

            AMREX_D_TERM(
            amrex::Real xBC = xhi                            ;,
            amrex::Real yBC = ylo + (amrex::Real(j) + 0.5)*dy;,
            amrex::Real zBC = zlo + (amrex::Real(k) + 0.5)*dz);

            amrex::Real BC_Dirichlet = BC_hi_Dirichlet[ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Neumann   = BC_hi_Neumann  [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Function  = BC_hi_Function [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real CellSize     = dx;

            amrex::Real f_Ghost;
            Ghost_Value(f_Ghost,
                        fInterior,
                        BC_Dirichlet, BC_Neumann, BC_Function,
                        CellSize);
            fArray(i,j,k) = f_Ghost;
        }
    }
#if (AMREX_SPACEDIM > 1)
    ID_Dimension = 1;
    if (j < jlo){
        if (BC_lo[ID_Dimension] != BoundaryConditionType::Periodic){
            int iInterior = i;
            int jInterior = 2*jlo - j - 1;
            int kInterior = k;
            amrex::Real fInterior = fArray(iInterior, jInterior, kInterior);

            AMREX_D_TERM(
            amrex::Real xBC = xlo + (amrex::Real(i) + 0.5)*dx;,
            amrex::Real yBC = ylo                            ;,
            amrex::Real zBC = zlo + (amrex::Real(k) + 0.5)*dz);

            amrex::Real BC_Dirichlet = BC_lo_Dirichlet[ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Neumann   = BC_lo_Neumann  [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Function  = BC_lo_Function [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real CellSize     = dy;

            amrex::Real f_Ghost;
            Ghost_Value(f_Ghost,
                        fInterior,
                        BC_Dirichlet, BC_Neumann, BC_Function,
                        CellSize);
            fArray(i,j,k) = f_Ghost;
        }
    }
    else if (j > jhi){
        if (BC_hi[ID_Dimension] != BoundaryConditionType::Periodic){
            int iInterior = i;
            int jInterior = 2*jhi - j + 1;
            int kInterior = k;
            amrex::Real fInterior = fArray(iInterior, jInterior, kInterior);

            AMREX_D_TERM(
            amrex::Real xBC = xlo + (amrex::Real(i) + 0.5)*dx;,
            amrex::Real yBC = yhi                            ;,
            amrex::Real zBC = zlo + (amrex::Real(k) + 0.5)*dz);

            amrex::Real BC_Dirichlet = BC_hi_Dirichlet[ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Neumann   = BC_hi_Neumann  [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Function  = BC_hi_Function [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real CellSize     = dy;

            amrex::Real f_Ghost;
            Ghost_Value(f_Ghost,
                        fInterior,
                        BC_Dirichlet, BC_Neumann, BC_Function,
                        CellSize);
            fArray(i,j,k) = f_Ghost;
        }
    }
#endif
#if (AMREX_SPACEDIM > 2)
    ID_Dimension = 2;
    if (k < klo){
        if (BC_lo[ID_Dimension] != BoundaryConditionType::Periodic){
            int iInterior = i;
            int jInterior = j;
            int kInterior = 2*klo - k - 1;
            amrex::Real fInterior = fArray(iInterior, jInterior, kInterior);

            AMREX_D_TERM(
            amrex::Real xBC = xlo + (amrex::Real(i) + 0.5)*dx;,
            amrex::Real yBC = ylo + (amrex::Real(j) + 0.5)*dy;,
            amrex::Real zBC = zlo                            );

            amrex::Real BC_Dirichlet = BC_lo_Dirichlet[ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Neumann   = BC_lo_Neumann  [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Function  = BC_lo_Function [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real CellSize     = dz;

            amrex::Real f_Ghost;
            Ghost_Value(f_Ghost,
                        fInterior,
                        BC_Dirichlet, BC_Neumann, BC_Function,
                        CellSize);
            fArray(i,j,k) = f_Ghost;
        }
    }
    else if (k > khi){
        if (BC_hi[ID_Dimension] != BoundaryConditionType::Periodic){
            int iInterior = i;
            int jInterior = j;
            int kInterior = 2*khi - k + 1;
            amrex::Real fInterior = fArray(iInterior, jInterior, kInterior);

            AMREX_D_TERM(
            amrex::Real xBC = xlo + (amrex::Real(i) + 0.5)*dx;,
            amrex::Real yBC = ylo + (amrex::Real(j) + 0.5)*dy;,
            amrex::Real zBC = zhi                            );

            amrex::Real BC_Dirichlet = BC_hi_Dirichlet[ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Neumann   = BC_hi_Neumann  [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real BC_Function  = BC_hi_Function [ID_Dimension](AMREX_D_DECL(xBC, yBC, zBC), time);
            amrex::Real CellSize     = dz;

            amrex::Real f_Ghost;
            Ghost_Value(f_Ghost,
                        fInterior,
                        BC_Dirichlet, BC_Neumann, BC_Function,
                        CellSize);
            fArray(i,j,k) = f_Ghost;
        }
    }
#endif
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Interpolation2Face_Linear (int i, int j, int k,
                                amrex::Real& f_Interpolation,
                                amrex::Array4<amrex::Real const> const& f,
                                int ID_Dimension)
{
    if (ID_Dimension == 0){
        f_Interpolation = 0.5*(f(i,j,k) + f(i-1,j,k)); // f_{i-1/2,j,k}
    }
#if (AMREX_SPACEDIM > 1)
    else if (ID_Dimension == 1){
        f_Interpolation = 0.5*(f(i,j,k) + f(i,j-1,k)); // f_{i,j-1/2,k}
    }
#endif
#if (AMREX_SPACEDIM > 2)
    else if (ID_Dimension == 2){
        f_Interpolation = 0.5*(f(i,j,k) + f(i,j,k-1)); // f_{i,j,k-1/2}
    }
#endif
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Gradient_Face (int i, int j, int k,
                    amrex::Real& Grad_f,
                    amrex::Array4<amrex::Real const> const& f,
                    amrex::Real const& CellSize,
                    int ID_Dimension)
{
    if (ID_Dimension == 0){
        Grad_f = (f(i,j,k) - f(i-1,j,k))/CellSize;
    }
#if (AMREX_SPACEDIM > 1)
    else if (ID_Dimension == 1){
        Grad_f = (f(i,j,k) - f(i,j-1,k))/CellSize;
    }
#endif
#if (AMREX_SPACEDIM > 2)
    else if (ID_Dimension == 2){
        Grad_f = (f(i,j,k) - f(i,j,k-1))/CellSize;
    }
#endif
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Gradient_Center (int i, int j, int k,
                      amrex::Real& Grad_f,
                      amrex::Array4<amrex::Real const> const& f,
                      amrex::Real const& CellSize,
                      int ID_Dimension)
{
    if (ID_Dimension == 0){
        Grad_f = (f(i+1,j,k) - f(i-1,j,k))/(2.0*CellSize);
    }
#if (AMREX_SPACEDIM > 1)
    else if (ID_Dimension == 1){
        Grad_f = (f(i,j+1,k) - f(i,j-1,k))/(2.0*CellSize);
    }
#endif
#if (AMREX_SPACEDIM > 2)
    else if (ID_Dimension == 2){
        Grad_f = (f(i,j,k+1) - f(i,j,k-1))/(2.0*CellSize);
    }
#endif
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Advection (int i, int j, int k,
                amrex::Real& Advection_f,
                amrex::Array4<amrex::Real const> const& f,
   AMREX_D_DECL(amrex::Array4<amrex::Real const> const& u,
                amrex::Array4<amrex::Real const> const& v,
                amrex::Array4<amrex::Real const> const& w),
   AMREX_D_DECL(amrex::Real const& dx,
                amrex::Real const& dy,
                amrex::Real const& dz),
                int ID_VelocityCenter)
{
    int ID_Dimension = 0;
    amrex::Real u_xFace_plus;
    amrex::Real u_xFace_mins;
    if (ID_VelocityCenter){
        Interpolation2Face_Linear(i+1, j, k,
                                  u_xFace_plus,
                                  u,
                                  ID_Dimension);
        Interpolation2Face_Linear(i, j, k,
                                  u_xFace_mins,
                                  u,
                                  ID_Dimension);
    }
    else{
        u_xFace_plus = u(i+1,j,k);
        u_xFace_mins = u(i  ,j,k);
    }
    
    amrex::Real f_xFace_plus;
    Interpolation2Face_Linear(i+1, j, k,
                              f_xFace_plus,
                              f,
                              ID_Dimension);
    amrex::Real f_xFace_mins;
    Interpolation2Face_Linear(i, j, k,
                              f_xFace_mins,
                              f,
                              ID_Dimension);
    amrex::Real duf_dx = (u_xFace_plus*f_xFace_plus - u_xFace_mins*f_xFace_mins)/dx;
#if (AMREX_SPACEDIM > 1)
    ID_Dimension = 1;
    amrex::Real v_yFace_plus;
    amrex::Real v_yFace_mins;
    if (ID_VelocityCenter){
        Interpolation2Face_Linear(i, j+1, k,
                                  v_yFace_plus,
                                  v,
                                  ID_Dimension);
        Interpolation2Face_Linear(i, j, k,
                                  v_yFace_mins,
                                  v,
                                  ID_Dimension);
    }
    else{
        v_yFace_plus = v(i,j+1,k);
        v_yFace_mins = v(i,j  ,k);
    }

    amrex::Real f_yFace_plus;
    Interpolation2Face_Linear(i, j+1, k,
                              f_yFace_plus,
                              f,
                              ID_Dimension);
    amrex::Real f_yFace_mins;
    Interpolation2Face_Linear(i, j, k,
                              f_yFace_mins,
                              f,
                              ID_Dimension);
    amrex::Real dvf_dy = (v_yFace_plus*f_yFace_plus - v_yFace_mins*f_yFace_mins)/dy;
#endif
#if (AMREX_SPACEDIM > 2)
    ID_Dimension = 2;
    amrex::Real w_zFace_plus;
    amrex::Real w_zFace_mins;
    if (ID_VelocityCenter){
        Interpolation2Face_Linear(i, j, k+1,
                                  w_zFace_plus,
                                  w,
                                  ID_Dimension);
        Interpolation2Face_Linear(i, j, k,
                                  w_zFace_mins,
                                  w,
                                  ID_Dimension);
    }
    else{
        w_zFace_plus = w(i,j,k+1);
        w_zFace_mins = w(i,j,k  );
    }

    amrex::Real f_zFace_plus;
    Interpolation2Face_Linear(i, j, k+1,
                              f_zFace_plus,
                              f,
                              ID_Dimension);
    amrex::Real f_zFace_mins;
    Interpolation2Face_Linear(i, j, k,
                              f_zFace_mins,
                              f,
                              ID_Dimension);
    amrex::Real dwf_dz = (w_zFace_plus*f_zFace_plus - w_zFace_mins*f_zFace_mins)/dz;
#endif
    Advection_f = AMREX_D_TERM(duf_dx, + dvf_dy, + dwf_dz);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Divergence (int i, int j, int k,
                 amrex::Real& Divergence_f,
    AMREX_D_DECL(amrex::Array4<amrex::Real const> const& fx,
                 amrex::Array4<amrex::Real const> const& fy,
                 amrex::Array4<amrex::Real const> const& fz),
    AMREX_D_DECL(amrex::Real const& dx,
                 amrex::Real const& dy,
                 amrex::Real const& dz),
                 int ID_FluxCenter)
{
    int ID_Dimension = 0;
    amrex::Real fx_plus;
    amrex::Real fx_mins;
    if (ID_FluxCenter){
        Interpolation2Face_Linear(i+1, j, k,
                                  fx_plus,
                                  fx,
                                  ID_Dimension);
        Interpolation2Face_Linear(i, j, k,
                                  fx_mins,
                                  fx,
                                  ID_Dimension);
    }
    else{
        fx_plus = fx(i+1,j,k);
        fx_mins = fx(i  ,j,k);
    }
    amrex::Real dfx_dx = (fx_plus - fx_mins)/dx;
#if (AMREX_SPACEDIM > 1)
    ID_Dimension = 1;
    amrex::Real fy_plus;
    amrex::Real fy_mins;
    if (ID_FluxCenter){
        Interpolation2Face_Linear(i, j+1, k,
                                  fy_plus,
                                  fy,
                                  ID_Dimension);
        Interpolation2Face_Linear(i, j, k,
                                  fy_mins,
                                  fy,
                                  ID_Dimension);
    }
    else{
        fy_plus = fy(i,j+1,k);
        fy_mins = fy(i,j  ,k);
    }
    amrex::Real dfy_dy = (fy_plus - fy_mins)/dy;
#endif
#if (AMREX_SPACEDIM > 2)
    ID_Dimension = 2;
    amrex::Real fz_plus;
    amrex::Real fz_mins;
    if (ID_FluxCenter){
        Interpolation2Face_Linear(i, j, k+1,
                                  fz_plus,
                                  fz,
                                  ID_Dimension);
        Interpolation2Face_Linear(i, j, k,
                                  fz_mins,
                                  fz,
                                  ID_Dimension);
    }
    else{
        fz_plus = fz(i,j,k+1);
        fz_mins = fz(i,j,k  );
    }
    amrex::Real dfz_dz = (fz_plus - fz_mins)/dz;
#endif
    Divergence_f = AMREX_D_TERM(dfx_dx, + dfy_dy, + dfz_dz);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Laplacian (int i, int j, int k,
                amrex::Real& Laplacian_f,
                amrex::Array4<amrex::Real const> const& f,
   AMREX_D_DECL(amrex::Real const& dx,
                amrex::Real const& dy,
                amrex::Real const& dz))
{
    int ID_Dimension = 0;
    amrex::Real df_dx_xFace_plus;
    Gradient_Face(i+1, j, k,
                  df_dx_xFace_plus,
                  f,
                  dx,
                  ID_Dimension);
    amrex::Real df_dx_xFace_mins;
    Gradient_Face(i, j, k,
                  df_dx_xFace_mins,
                  f,
                  dx,
                  ID_Dimension);
    amrex::Real d2f_dx2 = (df_dx_xFace_plus - df_dx_xFace_mins)/dx;
#if (AMREX_SPACEDIM > 1)
    ID_Dimension = 1;
    amrex::Real df_dy_yFace_plus;
    Gradient_Face(i, j+1, k,
                  df_dy_yFace_plus,
                  f,
                  dy,
                  ID_Dimension);
    amrex::Real df_dy_yFace_mins;
    Gradient_Face(i, j, k,
                  df_dy_yFace_mins,
                  f,
                  dy,
                  ID_Dimension);
    amrex::Real d2f_dy2 = (df_dy_yFace_plus - df_dy_yFace_mins)/dy;
#endif
#if (AMREX_SPACEDIM > 2)
    ID_Dimension = 2;
    amrex::Real df_dz_zFace_plus;
    Gradient_Face(i, j, k+1,
                  df_dz_zFace_plus,
                  f,
                  dz,
                  ID_Dimension);
    amrex::Real df_dz_zFace_mins;
    Gradient_Face(i, j, k,
                  df_dz_zFace_mins,
                  f,
                  dz,
                  ID_Dimension);
    amrex::Real d2f_dz2 = (df_dz_zFace_plus - df_dz_zFace_mins)/dz;
#endif
    Laplacian_f = AMREX_D_TERM(d2f_dx2, + d2f_dy2, + d2f_dz2);
}
#endif
