#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include "ProblemSetup.H"

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Initial_Condition (int i, int j, int k,
                        AMREX_D_DECL(
                        amrex::Array4<amrex::Real> const& uArray,
                        amrex::Array4<amrex::Real> const& vArray,
                        amrex::Array4<amrex::Real> const& wArray),
                        amrex::Array4<amrex::Real> const& PArray,
                        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dX,
                        amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                        InitialConditionData const& initial_condition_data,
                        amrex::RandomEngine const& random_engine)
{
    AMREX_D_TERM(amrex::Real x = prob_lo[0] + (amrex::Real(i) + 0.5) * dX[0];,
                 amrex::Real y = prob_lo[1] + (amrex::Real(j) + 0.5) * dX[1];,
                 amrex::Real z = prob_lo[2] + (amrex::Real(k) + 0.5) * dX[2]);

    amrex::Real AMREX_D_DECL(u, v, w), P;
    Initial_Condition(AMREX_D_DECL(u, v, w), P,
                      AMREX_D_DECL(x, y, z),
                      initial_condition_data);

    amrex::Real u_Random = amrex::RandomNormal(0.0, initial_condition_data.u_noise, random_engine);
#if (AMREX_SPACEDIM > 1)
    amrex::Real v_Random = amrex::RandomNormal(0.0, initial_condition_data.v_noise, random_engine);
#endif
#if (AMREX_SPACEDIM > 2)
    amrex::Real w_Random = amrex::RandomNormal(0.0, initial_condition_data.w_noise, random_engine);
#endif
    amrex::Real P_Random = amrex::RandomNormal(0.0, initial_condition_data.P_noise, random_engine);

    AMREX_D_TERM(
    uArray(i,j,k) = u + u_Random;,
    vArray(i,j,k) = v + v_Random;,
    wArray(i,j,k) = w + w_Random);
    PArray(i,j,k) = P + P_Random;
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
            int iInterior = ilo;
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
            int iInterior = ihi;
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
            int jInterior = jlo;
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
            int jInterior = jhi;
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
            int kInterior = klo;
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
            int kInterior = khi;
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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Velocity_Update (int i, int j, int k,
                      amrex::Array4<amrex::Real> const& Vel_new,
                      amrex::Array4<amrex::Real const> const& Vel_old,
         AMREX_D_DECL(amrex::Array4<amrex::Real const> const& u_old,
                      amrex::Array4<amrex::Real const> const& v_old,
                      amrex::Array4<amrex::Real const> const& w_old),
                      amrex::Array4<amrex::Real const> const& P_old,
         AMREX_D_DECL(amrex::Real const& dx,
                      amrex::Real const& dy,
                      amrex::Real const& dz),
                      amrex::Real const& dt,
         AMREX_D_DECL(amrex::Real const& x,
                      amrex::Real const& y,
                      amrex::Real const& z),
                      amrex::Real const& time,
                      amrex::Real const& nu,
                      ImmersedBoundaryData const& immersed_boundary_data,
                      int ID_VelocityCenter,
                      int ID_Dimension)
{
    amrex::Array<amrex::Real,AMREX_SPACEDIM> dX = {AMREX_D_DECL(dx, dy, dz)};
    amrex::Real Advection_Vel;
    Advection(i, j, k,
              Advection_Vel,
              Vel_old,
 AMREX_D_DECL(u_old, v_old, w_old),
 AMREX_D_DECL(   dx,    dy,    dz),
              ID_VelocityCenter);

    amrex::Real Laplacian_Vel;
    Laplacian(i, j, k,
              Laplacian_Vel,
              Vel_old,
 AMREX_D_DECL(dx, dy, dz));

    amrex::Real Grad_P;
    Gradient_Center(i, j, k,
                    Grad_P,
                    P_old, dX[ID_Dimension], ID_Dimension);

    amrex::Real Alpha_Solid;
    Solid_Fraction(Alpha_Solid,
      AMREX_D_DECL(x, y, z), time+dt,
                   immersed_boundary_data);

    amrex::Array<amrex::Real,AMREX_SPACEDIM> Vel_Solid;
    Solid_Velocity(Vel_Solid,
      AMREX_D_DECL(x, y, z), time+dt,
                   immersed_boundary_data);

    amrex::Real Cd;
    Drag_Coefficient(Cd,
                     Alpha_Solid, dt);

    amrex::Real Force = 0.0;

    Vel_new(i,j,k) = (Vel_old(i,j,k) + dt*(-Advection_Vel - Grad_P + nu*Laplacian_Vel + Force) + dt*Cd*Vel_Solid[ID_Dimension])/(1.0 + dt*Cd);
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void Velocity_Correct (int i, int j, int k,
                      amrex::Array4<amrex::Real> const& Vel_new,
                      amrex::Array4<amrex::Real const> const& Vel_old,
                      amrex::Array4<amrex::Real const> const& PCorrect,
         AMREX_D_DECL(amrex::Real const& dx,
                      amrex::Real const& dy,
                      amrex::Real const& dz),
                      amrex::Real const& dt,
                      int ID_Dimension)
{
    amrex::Array<amrex::Real,AMREX_SPACEDIM> dX = {AMREX_D_DECL(dx, dy, dz)};
    amrex::Real Grad_PCorrect;
    Gradient_Center(i, j, k,
                    Grad_PCorrect,
                    PCorrect, dX[ID_Dimension], ID_Dimension);
    Vel_new(i,j,k) = Vel_old(i,j,k) - dt*Grad_PCorrect;
}
#endif
