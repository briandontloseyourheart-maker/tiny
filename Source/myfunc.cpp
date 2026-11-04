
#include "myfunc.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void advance (MultiFab& u_old, MultiFab& u_new, BoundaryCondition const& BoundaryCondition_u,
#if (AMREX_SPACEDIM > 1)
              MultiFab& v_old, MultiFab& v_new, BoundaryCondition const& BoundaryCondition_v,
#endif
#if (AMREX_SPACEDIM > 2)
              MultiFab& w_old, MultiFab& w_new, BoundaryCondition const& BoundaryCondition_w,
#endif
              MultiFab& P_old, MultiFab& P_new, BoundaryCondition const& BoundaryCondition_P,
              Array<MultiFab,AMREX_SPACEDIM>& Vel_old, Array<MultiFab,AMREX_SPACEDIM>& Vel_new,
              Array<Real,AMREX_SPACEDIM>& Force_Hydrodynamic,
              Geometry const& geom,
              Real const& dt,
              Real const& time,
              Real const& nu,
              ImmersedBoundary const& immersed_boundary,
              Real const& tol_rel,
              int max_iter,
              int max_fmg_iter,
              int verbose,
              int bottom_verbose)
{
    AMREX_D_TERM(const Real dx = geom.CellSize(0);,
                 const Real dy = geom.CellSize(1);,
                 const Real dz = geom.CellSize(2));
    const Array<const Real,AMREX_SPACEDIM> dX = {AMREX_D_DECL(dx, dy, dz)};

    AMREX_D_TERM(const Real xlo = geom.ProbLo(0);,
                 const Real ylo = geom.ProbLo(1);,
                 const Real zlo = geom.ProbLo(2));
    const Array<const Real,AMREX_SPACEDIM> prob_lo = {AMREX_D_DECL(xlo, ylo, zlo)};

    // Velocity update
    Boundary_Condition(u_old, BoundaryCondition_u, geom, time);
#if (AMREX_SPACEDIM > 1)
    Boundary_Condition(v_old, BoundaryCondition_v, geom, time);
#endif
#if (AMREX_SPACEDIM > 2)
    Boundary_Condition(w_old, BoundaryCondition_w, geom, time);
#endif
    Boundary_Condition(P_old, BoundaryCondition_P, geom, time);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& uf_oldArray = Vel_old[0].const_array(mfi);
        auto const& u_oldArray  = u_old.const_array(mfi);
        auto const& u_newArray  = u_new.array(mfi);
#if (AMREX_SPACEDIM > 1)
        auto const& vf_oldArray = Vel_old[1].const_array(mfi);
        auto const& v_oldArray  = v_old.const_array(mfi);
        auto const& v_newArray  = v_new.array(mfi);
#endif
#if (AMREX_SPACEDIM > 2)
        auto const& wf_oldArray = Vel_old[2].const_array(mfi);
        auto const& w_oldArray  = w_old.const_array(mfi);
        auto const& w_newArray  = w_new.array(mfi);
#endif
        auto const& P_oldArray = P_old.const_array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            AMREX_D_TERM(amrex::Real x = xlo + (amrex::Real(i) + 0.5) * dx;,
                         amrex::Real y = ylo + (amrex::Real(j) + 0.5) * dy;,
                         amrex::Real z = zlo + (amrex::Real(k) + 0.5) * dz);

            int ID_VelocityCenter = 0;
            int ID_Dimension = 0;
            Velocity_Update(i, j, k,
                            u_newArray,
                            u_oldArray,
               AMREX_D_DECL(uf_oldArray, vf_oldArray, wf_oldArray), P_oldArray,
               AMREX_D_DECL(dx, dy, dz), dt,
               AMREX_D_DECL( x,  y,  z), time,
                            nu,
                            immersed_boundary,
                            ID_VelocityCenter,
                            ID_Dimension);
#if (AMREX_SPACEDIM > 1)
            ID_Dimension = 1;
            Velocity_Update(i, j, k,
                            v_newArray,
                            v_oldArray,
               AMREX_D_DECL(uf_oldArray, vf_oldArray, wf_oldArray), P_oldArray,
               AMREX_D_DECL(dx, dy, dz), dt,
               AMREX_D_DECL( x,  y,  z), time,
                            nu,
                            immersed_boundary,
                            ID_VelocityCenter,
                            ID_Dimension);
#endif
#if (AMREX_SPACEDIM > 2)
            ID_Dimension = 2;
            Velocity_Update(i, j, k,
                            w_newArray,
                            w_oldArray,
               AMREX_D_DECL(uf_oldArray, vf_oldArray, wf_oldArray), P_oldArray,
               AMREX_D_DECL(dx, dy, dz), dt,
               AMREX_D_DECL( x,  y,  z), time,
                            nu,
                            immersed_boundary,
                            ID_VelocityCenter,
                            ID_Dimension);
#endif
        });
    }

    // Compute Force_Hydrodynamic
    MultiFab Force_Hydro(u_new.boxArray(), u_new.DistributionMap(), AMREX_SPACEDIM, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Force_Hydro, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& u_newArray  = u_new.const_array(mfi);
#if (AMREX_SPACEDIM > 1)
        auto const& v_newArray  = v_new.const_array(mfi);
#endif
#if (AMREX_SPACEDIM > 2)
        auto const& w_newArray  = w_new.const_array(mfi);
#endif
        auto const& Force_HydroArray = Force_Hydro.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            AMREX_D_TERM(amrex::Real x = xlo + (amrex::Real(i) + 0.5) * dx;,
                         amrex::Real y = ylo + (amrex::Real(j) + 0.5) * dy;,
                         amrex::Real z = zlo + (amrex::Real(k) + 0.5) * dz);

            Real Alpha_Solid;
            Solid_Fraction(Alpha_Solid,
              AMREX_D_DECL(x, y, z), time+dt,
                           immersed_boundary);

            Array<Real,AMREX_SPACEDIM> Vel_Solid;
            Solid_Velocity(Vel_Solid,
              AMREX_D_DECL(x, y, z), time+dt,
                           immersed_boundary);

            Real Cd;
            Drag_Coefficient(Cd,
                             Alpha_Solid, dt);

            Force_HydroArray(i,j,k,0) = -Cd*(Vel_Solid[0] - u_newArray(i,j,k));
#if (AMREX_SPACEDIM > 1)
            Force_HydroArray(i,j,k,1) = -Cd*(Vel_Solid[1] - v_newArray(i,j,k));
#endif
#if (AMREX_SPACEDIM > 2)
            Force_HydroArray(i,j,k,2) = -Cd*(Vel_Solid[2] - w_newArray(i,j,k));
#endif
        });
    }
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
        Force_Hydrodynamic[idim] = Force_Hydro.sum(idim)*(AMREX_D_TERM(dx,*dy,*dz));
    }

    // Remove pressure gradient
    MultiFab::Copy(u_old, u_new, 0, 0, 1, 0);
#if (AMREX_SPACEDIM > 1)
    MultiFab::Copy(v_old, v_new, 0, 0, 1, 0);
#endif
#if (AMREX_SPACEDIM > 2)
    MultiFab::Copy(w_old, w_new, 0, 0, 1, 0);
#endif
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& u_oldArray = u_old.const_array(mfi);
        auto const& u_newArray = u_new.array(mfi);
#if (AMREX_SPACEDIM > 1)
        auto const& v_oldArray = v_old.const_array(mfi);
        auto const& v_newArray = v_new.array(mfi);
#endif
#if (AMREX_SPACEDIM > 2)
        auto const& w_oldArray = w_old.const_array(mfi);
        auto const& w_newArray = w_new.array(mfi);
#endif
        auto const& P_oldArray = P_old.const_array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int ID_Dimension = 0;
            Real Grad_Pold;
            Gradient_Center(i, j, k,
                            Grad_Pold,
                            P_oldArray, dX[ID_Dimension], ID_Dimension);
            u_newArray(i,j,k) = u_oldArray(i,j,k) + dt*Grad_Pold;
#if (AMREX_SPACEDIM > 1)
            ID_Dimension = 1;
            Gradient_Center(i, j, k,
                            Grad_Pold,
                            P_oldArray, dX[ID_Dimension], ID_Dimension);
            v_newArray(i,j,k) = v_oldArray(i,j,k) + dt*Grad_Pold;
#endif
#if (AMREX_SPACEDIM > 2)
            ID_Dimension = 2;
            Gradient_Center(i, j, k,
                            Grad_Pold,
                            P_oldArray, dX[ID_Dimension], ID_Dimension);
            w_newArray(i,j,k) = w_oldArray(i,j,k) + dt*Grad_Pold;
#endif
        });
    }

    // Compute Div_Vel
    MultiFab::Copy(u_old, u_new, 0, 0, 1, 0);
    Boundary_Condition(u_old, BoundaryCondition_u, geom, time+dt);
#if (AMREX_SPACEDIM > 1)
    MultiFab::Copy(v_old, v_new, 0, 0, 1, 0);
    Boundary_Condition(v_old, BoundaryCondition_v, geom, time+dt);
#endif
#if (AMREX_SPACEDIM > 2)
    MultiFab::Copy(w_old, w_new, 0, 0, 1, 0);
    Boundary_Condition(w_old, BoundaryCondition_w, geom, time+dt);
#endif
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& P_oldArray = P_old.const_array(mfi);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            Array4<Real const> Vel_oldArray;
            if (idim == 0){
                Vel_oldArray = u_old.const_array(mfi);
            }
#if (AMREX_SPACEDIM > 1)
            else if (idim == 1){
                Vel_oldArray = v_old.const_array(mfi);
            }
#endif
#if (AMREX_SPACEDIM > 2)
            else if (idim == 2){
                Vel_oldArray = w_old.const_array(mfi);
            }
#endif
            auto const& Vel_newArray = Vel_new[idim].array(mfi);

            const Box& bx = mfi.nodaltilebox(idim);
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Real Vel_Interpolation;
                Interpolation2Face_Linear(i, j, k,
                                          Vel_Interpolation,
                                          Vel_oldArray,
                                          idim);
                Real Grad_Pold;
                Gradient_Face(i, j, k,
                              Grad_Pold,
                              P_oldArray, dX[idim], idim);
                Vel_newArray(i,j,k) = Vel_Interpolation - dt*Grad_Pold;
            });
        }
    }

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
        MultiFab::Copy(Vel_old[idim], Vel_new[idim], 0, 0, 1, 0);
    }
    MultiFab Div_Vel(P_old.boxArray(), P_old.DistributionMap(), 1, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(Div_Vel, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& uf_oldArray = Vel_old[0].const_array(mfi);
#if (AMREX_SPACEDIM > 1)
        auto const& vf_oldArray = Vel_old[1].const_array(mfi);
#endif
#if (AMREX_SPACEDIM > 2)
        auto const& wf_oldArray = Vel_old[2].const_array(mfi);
#endif
        auto const& Div_VelArray = Div_Vel.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int ID_FluxCenter = 0;
            Real Divergence_Vel;
            Divergence(i, j, k,
                       Divergence_Vel,
          AMREX_D_DECL(uf_oldArray, vf_oldArray, wf_oldArray),
          AMREX_D_DECL(         dx,          dy,          dz),
                       ID_FluxCenter);
            Div_VelArray(i,j,k) = Divergence_Vel;
        });
    }

    // Pressure correction
    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class (ascalar*acoef - bscalar div bcoef grad) phi = RHS
    MLABecLaplacian mlabec({geom}, {Div_Vel.boxArray()}, {Div_Vel.DistributionMap()}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    Array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
    Array<LinOpBCType,AMREX_SPACEDIM> bc_hi;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (BoundaryCondition_P.lo[idim] == BoundaryConditionType::Periodic) {
            bc_lo[idim] = LinOpBCType::Periodic;
        }
        else if (BoundaryCondition_P.lo[idim] == BoundaryConditionType::Neumann) {
            bc_lo[idim] = LinOpBCType::Neumann;
        }
        else if (BoundaryCondition_P.lo[idim] == BoundaryConditionType::Dirichlet) {
            bc_lo[idim] = LinOpBCType::Dirichlet;
        }
        else {
            amrex::Abort("Invalid bc_lo");
        }

        // hi-side BCs
        if (BoundaryCondition_P.hi[idim] == BoundaryConditionType::Periodic) {
            bc_hi[idim] = LinOpBCType::Periodic;
        }
        else if (BoundaryCondition_P.hi[idim] == BoundaryConditionType::Neumann) {
            bc_hi[idim] = LinOpBCType::Neumann;
        }
        else if (BoundaryCondition_P.hi[idim] == BoundaryConditionType::Dirichlet) {
            bc_hi[idim] = LinOpBCType::Dirichlet;
        }
        else {
            amrex::Abort("Invalid bc_hi");
        }
    }
    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(bc_lo, bc_hi);

    // set the boundary conditions
    mlabec.setLevelBC(0, nullptr);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -dt;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(Div_Vel.boxArray(), Div_Vel.DistributionMap(), 1, 0);
    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(0.0);
    mlabec.setACoeffs(0, acoef);

    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in bcoef MultiFabs and load them into the solver.
    Array<MultiFab,AMREX_SPACEDIM> bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        BoxArray ba = acoef.boxArray();
        ba.surroundingNodes(idim);
        bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
        bcoef[idim].setVal(1.0);
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));

    // build an MLMG solver
    MLMG mlmg(mlabec);

    // set solver parameters
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
//    mlmg.setBottomSolver(MLMG::BottomSolver::smoother);

    // relative and absolute tolerances for linear solve
    const Real tol_abs = 0.0;

    // Solve linear system
    MultiFab PCorrect(Div_Vel.boxArray(), Div_Vel.DistributionMap(), 1, P_old.nGrow());
    PCorrect.setVal(0.0);
    mlmg.solve({&PCorrect}, {&Div_Vel}, tol_rel, tol_abs);

    // Velocity correction
    Boundary_Condition(PCorrect, BoundaryCondition_P, geom, time+dt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& PCorrect_Array = PCorrect.const_array(mfi);
        auto const& P_oldArray = P_old.const_array(mfi);
        auto const& P_newArray = P_new.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            P_newArray(i,j,k) = P_oldArray(i,j,k) + PCorrect_Array(i,j,k);
        });

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            auto const& Vel_oldArray = Vel_old[idim].const_array(mfi);
            auto const& Vel_newArray = Vel_new[idim].array(mfi);

            const Box& bx_face = mfi.nodaltilebox(idim);
            amrex::ParallelFor(bx_face,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Real Grad_PCorrect;
                Gradient_Face(i, j, k,
                              Grad_PCorrect,
                              PCorrect_Array, dX[idim], idim);
                Vel_newArray(i,j,k) = Vel_oldArray(i,j,k) - dt*Grad_PCorrect;
            });
        }
    }

    Boundary_Condition(P_new, BoundaryCondition_P, geom, time+dt);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& u_oldArray = u_old.const_array(mfi);
        auto const& u_newArray = u_new.array(mfi);
#if (AMREX_SPACEDIM > 1)
        auto const& v_oldArray = v_old.const_array(mfi);
        auto const& v_newArray = v_new.array(mfi);
#endif
#if (AMREX_SPACEDIM > 2)
        auto const& w_oldArray = w_old.const_array(mfi);
        auto const& w_newArray = w_new.array(mfi);
#endif
        auto const& P_newArray = P_new.const_array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            int ID_Dimension = 0;
            Velocity_Correct(i, j, k,
                             u_newArray,
                             u_oldArray,
                             P_newArray,
                AMREX_D_DECL(dx, dy, dz), dt,
                             ID_Dimension);
#if (AMREX_SPACEDIM > 1)
            ID_Dimension = 1;
            Velocity_Correct(i, j, k,
                             v_newArray,
                             v_oldArray,
                             P_newArray,
                AMREX_D_DECL(dx, dy, dz), dt,
                             ID_Dimension);
#endif
#if (AMREX_SPACEDIM > 2)
            ID_Dimension = 2;
            Velocity_Correct(i, j, k,
                             w_newArray,
                             w_oldArray,
                             P_newArray,
                AMREX_D_DECL(dx, dy, dz), dt,
                             ID_Dimension);
#endif
        });
    }

}


void Initial_Condition (amrex::MultiFab& u_new, BoundaryCondition const& BoundaryCondition_u,
#if (AMREX_SPACEDIM > 1)
                        amrex::MultiFab& v_new, BoundaryCondition const& BoundaryCondition_v,
#endif
#if (AMREX_SPACEDIM > 2)
                        amrex::MultiFab& w_new, BoundaryCondition const& BoundaryCondition_w,
#endif
                        amrex::MultiFab& P_new,
                        amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& Vel_new,
                        amrex::Geometry const& geom,
                        InitialCondition const& initial_condition)
{
    GpuArray<Real,AMREX_SPACEDIM> dX      = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        AMREX_D_TERM(
        auto const& u_newArray = u_new.array(mfi);,
        auto const& v_newArray = v_new.array(mfi);,
        auto const& w_newArray = w_new.array(mfi));
        auto const& P_newArray = P_new.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const& random_engine)
        {
            Initial_Condition(i, j, k,
                 AMREX_D_DECL(u_newArray, v_newArray, w_newArray), P_newArray,
                              dX, prob_lo,
                              initial_condition,
                              random_engine);
        });
    }

    AMREX_D_TERM(
    Boundary_Condition(u_new, BoundaryCondition_u, geom, 0.0);,
    Boundary_Condition(v_new, BoundaryCondition_v, geom, 0.0);,
    Boundary_Condition(w_new, BoundaryCondition_w, geom, 0.0));
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(P_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            Array4<Real const> f_newArray;
            if (idim == 0){
                f_newArray = u_new.const_array(mfi);
            }
#if (AMREX_SPACEDIM > 1)
            else if (idim == 1){
                f_newArray = v_new.const_array(mfi);
            }
#endif
#if (AMREX_SPACEDIM > 2)
            else if (idim == 2){
                f_newArray = w_new.const_array(mfi);
            }
#endif
            auto const& Vel_newArray = Vel_new[idim].array(mfi);

            const Box& bx = mfi.nodaltilebox(idim);
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Real f_Interpolation;
                Interpolation2Face_Linear(i, j, k,
                                          f_Interpolation,
                                          f_newArray,
                                          idim);
                Vel_newArray(i,j,k) = f_Interpolation;
            });
        }
    }
}


void Boundary_Condition (amrex::MultiFab& f,
                         BoundaryCondition const& BoundaryCondition_f,
                         amrex::Geometry const& geom,
                         amrex::Real const& time)
{
    // Fill the ghost cells of each grid from the other grids includes periodic domain boundaries.
    f.FillBoundary(geom.periodicity());

    // Fill the ghost cells of non-periodic boundaries.
    int nGrow = f.nGrow();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();
    GpuArray<Real,AMREX_SPACEDIM> dX      = geom.CellSizeArray();

    const auto& domain_box = geom.Domain();
    const auto& domain_lo = domain_box.loVect();
    const auto& domain_hi = domain_box.hiVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(f, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.growntilebox(nGrow);
        auto const& fArray = f.array(mfi);
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Boundary_Condition(i, j, k,
                               fArray,
                               BoundaryCondition_f.lo, BoundaryCondition_f.hi,
                               BoundaryCondition_f.lo_Dirichlet, BoundaryCondition_f.hi_Dirichlet,
                               BoundaryCondition_f.lo_Neumann  , BoundaryCondition_f.hi_Neumann,
                               BoundaryCondition_f.lo_Function , BoundaryCondition_f.hi_Function,
                  AMREX_D_DECL(domain_lo[0], domain_lo[1], domain_lo[2]),
                  AMREX_D_DECL(domain_hi[0], domain_hi[1], domain_hi[2]),
                  AMREX_D_DECL(  prob_lo[0],   prob_lo[1],   prob_lo[2]),
                  AMREX_D_DECL(  prob_hi[0],   prob_hi[1],   prob_hi[2]),
                  AMREX_D_DECL(       dX[0],        dX[1],        dX[2]),
                               time);
        });
    }
}
