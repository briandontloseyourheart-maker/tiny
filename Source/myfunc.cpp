
#include "myfunc.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void advance (MultiFab& U_old, MultiFab& U_new,
              Array<BoundaryCondition,NSTATE> const& BoundaryCondition_V,
              Array<Real,AMREX_SPACEDIM>& Force_Hydrodynamic,
              Geometry const& geom,
              Real const& dt,
              Real const& time,
              Real const& gamma,
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

    Boundary_Condition(U_old, BoundaryCondition_V, geom, time, gamma);

    ImmersedBoundaryData const immersed_boundary_data = immersed_boundary.Data;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(U_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& U_oldArray  = U_old.const_array(mfi);
        auto const& U_newArray  = U_new.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            AMREX_D_TERM(amrex::Real x = xlo + (amrex::Real(i) + 0.5) * dx;,
                         amrex::Real y = ylo + (amrex::Real(j) + 0.5) * dy;,
                         amrex::Real z = zlo + (amrex::Real(k) + 0.5) * dz);

            int ID_Dimension;
            Array<Real,NSTATE> Div_Flux;
            Array<Real,NSTATE> Flux_m;
            Array<Real,NSTATE> Flux_p;
            Real cmax;

            ID_Dimension = 0;
            Conservative2FluxHyperbolic(i, j, k,
                                        Flux_m, cmax,
                                        U_oldArray,
                                        gamma,
                                        ID_Dimension);
            Conservative2FluxHyperbolic(i+1, j, k,
                                        Flux_p, cmax,
                                        U_oldArray,
                                        gamma,
                                        ID_Dimension);
            for (int iState = 0; iState < NSTATE; iState++){
                Div_Flux[iState] = (Flux_p[iState] - Flux_m[iState])/dx;
            }
#if (AMREX_SPACEDIM > 1)
            ID_Dimension = 1;
            Conservative2FluxHyperbolic(i, j, k,
                                        Flux_m, cmax,
                                        U_oldArray,
                                        gamma,
                                        ID_Dimension);
            Conservative2FluxHyperbolic(i, j+1, k,
                                        Flux_p, cmax,
                                        U_oldArray,
                                        gamma,
                                        ID_Dimension);
            for (int iState = 0; iState < NSTATE; iState++){
                Div_Flux[iState] += (Flux_p[iState] - Flux_m[iState])/dy;
            }
#endif
#if (AMREX_SPACEDIM > 2)
            ID_Dimension = 2;
            Conservative2FluxHyperbolic(i, j, k,
                                        Flux_m, cmax,
                                        U_oldArray,
                                        gamma,
                                        ID_Dimension);
            Conservative2FluxHyperbolic(i, j, k+1,
                                        Flux_p, cmax,
                                        U_oldArray,
                                        gamma,
                                        ID_Dimension);
            for (int iState = 0; iState < NSTATE; iState++){
                Div_Flux[iState] += (Flux_p[iState] - Flux_m[iState])/dz;
            }
#endif
            for (int iState = 0; iState < NSTATE; iState++){
                U_newArray(i,j,k,iState) = U_oldArray(i,j,k,iState) - dt*Div_Flux[iState];
            }
        });
    }
}


void Initial_Condition (amrex::MultiFab& U_new,
                        amrex::Geometry const& geom,
                        InitialCondition const& initial_condition,
                        amrex::Real const& gamma)
{
    GpuArray<Real,AMREX_SPACEDIM> dX      = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    InitialConditionData const initial_condition_data = initial_condition.Data;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(U_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& U_newArray = U_new.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const& random_engine)
        {
            Initial_Condition(i, j, k,
                              U_newArray,
                              dX, prob_lo,
                              initial_condition_data,
                              random_engine,
                              gamma);
        });
    }
}


void Boundary_Condition (amrex::MultiFab& U,
                         amrex::Array<BoundaryCondition,NSTATE> const& BoundaryCondition_V,
                         amrex::Geometry const& geom,
                         amrex::Real const& time,
                         amrex::Real const& gamma)
{
    // Fill the primitive variables
    amrex::MultiFab V(U.boxArray(), U.DistributionMap(), U.nComp(), U.nGrow());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(V, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& UArray = U.const_array(mfi);
        auto const& VArray = V.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Array<amrex::Real,NSTATE> Conservative;
            for (int iState = 0; iState < NSTATE; iState++){
                Conservative[iState] = UArray(i,j,k,iState);
            }

            amrex::Array<amrex::Real,NSTATE> Primitive;
            Conservative2Primitive(Primitive,
                                   Conservative,
                                   gamma);

            for (int iState = 0; iState < NSTATE; iState++){
                VArray(i,j,k,iState) = Primitive[iState];
            }
        });
    }

    // Fill the ghost cells of each grid from the other grids includes periodic domain boundaries.
    V.FillBoundary(geom.periodicity());

    // Fill the ghost cells of non-periodic boundaries.
    int nGrow = V.nGrow();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();
    GpuArray<Real,AMREX_SPACEDIM> dX      = geom.CellSizeArray();

    const auto& domain_box = geom.Domain();
    const auto& domain_lo = domain_box.smallEnd();
    const auto& domain_hi = domain_box.bigEnd();
    for (int iState = 0; iState < NSTATE; iState++){
        amrex::MultiFab f(V, amrex::make_alias, iState, 1);
        BoundaryConditionData const BoundaryConditionData_f = BoundaryCondition_V[iState].Data;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(f, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& fArray = f.array(mfi);

            const Box& gbx = mfi.growntilebox(nGrow);
            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Boundary_Condition(i, j, k,
                                   fArray,
                                   BoundaryConditionData_f.lo, BoundaryConditionData_f.hi,
                                   BoundaryConditionData_f.lo_Dirichlet, BoundaryConditionData_f.hi_Dirichlet,
                                   BoundaryConditionData_f.lo_Neumann  , BoundaryConditionData_f.hi_Neumann,
                                   BoundaryConditionData_f.lo_Function , BoundaryConditionData_f.hi_Function,
                      AMREX_D_DECL(domain_lo[0], domain_lo[1], domain_lo[2]),
                      AMREX_D_DECL(domain_hi[0], domain_hi[1], domain_hi[2]),
                      AMREX_D_DECL(  prob_lo[0],   prob_lo[1],   prob_lo[2]),
                      AMREX_D_DECL(  prob_hi[0],   prob_hi[1],   prob_hi[2]),
                      AMREX_D_DECL(       dX[0],        dX[1],        dX[2]),
                                   time);
            });
        }
    }

    // Fill the conservative varialbes with ghost cells
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(U, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& VArray = V.const_array(mfi);
        auto const& UArray = U.array(mfi);

        const Box& gbx = mfi.growntilebox(nGrow);
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Array<amrex::Real,NSTATE> Primitive;
            for (int iState = 0; iState < NSTATE; iState++){
                Primitive[iState] = VArray(i,j,k,iState);
            }

            amrex::Array<amrex::Real,NSTATE> Conservative;
            Primitive2Conservative(Conservative,
                                   Primitive,
                                   gamma);

            for (int iState = 0; iState < NSTATE; iState++){
                UArray(i,j,k,iState) = Conservative[iState];
            }
        });
    }
}


void Max_WaveSpeed (Array<Real,AMREX_SPACEDIM>& max_wave_speed,
                    MultiFab& U,
                    Array<BoundaryCondition,NSTATE> const& BoundaryCondition_V,
                    Geometry const& geom,
                    Real const& time,
                    Real const& gamma)
{
    AMREX_D_TERM(const Real dx = geom.CellSize(0);,
                 const Real dy = geom.CellSize(1);,
                 const Real dz = geom.CellSize(2));
    const Array<const Real,AMREX_SPACEDIM> dX = {AMREX_D_DECL(dx, dy, dz)};

    AMREX_D_TERM(const Real xlo = geom.ProbLo(0);,
                 const Real ylo = geom.ProbLo(1);,
                 const Real zlo = geom.ProbLo(2));
    const Array<const Real,AMREX_SPACEDIM> prob_lo = {AMREX_D_DECL(xlo, ylo, zlo)};

    Boundary_Condition(U, BoundaryCondition_V, geom, time, gamma);

    MultiFab WaveSpeed(U.boxArray(), U.DistributionMap(), AMREX_SPACEDIM, 0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(WaveSpeed, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        auto const& U_Array         = U.const_array(mfi);
        auto const& WaveSpeed_Array = WaveSpeed.array(mfi);

        const Box& bx = mfi.tilebox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            AMREX_D_TERM(amrex::Real x = xlo + (amrex::Real(i) + 0.5) * dx;,
                         amrex::Real y = ylo + (amrex::Real(j) + 0.5) * dy;,
                         amrex::Real z = zlo + (amrex::Real(k) + 0.5) * dz);

            int ID_Dimension;
            Array<Real,NSTATE> Flux;
            Real cmax_m;
            Real cmax_p;

            ID_Dimension = 0;
            Conservative2FluxHyperbolic(i, j, k,
                                        Flux, cmax_m,
                                        U_Array,
                                        gamma,
                                        ID_Dimension);
            Conservative2FluxHyperbolic(i+1, j, k,
                                        Flux, cmax_p,
                                        U_Array,
                                        gamma,
                                        ID_Dimension);
            WaveSpeed_Array(i,j,k,ID_Dimension) = std::max(cmax_m, cmax_p);
#if (AMREX_SPACEDIM > 1)
            ID_Dimension = 1;
            Conservative2FluxHyperbolic(i, j, k,
                                        Flux, cmax_m,
                                        U_Array,
                                        gamma,
                                        ID_Dimension);
            Conservative2FluxHyperbolic(i, j+1, k,
                                        Flux, cmax_p,
                                        U_Array,
                                        gamma,
                                        ID_Dimension);
            WaveSpeed_Array(i,j,k,ID_Dimension) = std::max(cmax_m, cmax_p);
#endif
#if (AMREX_SPACEDIM > 2)
            ID_Dimension = 2;
            Conservative2FluxHyperbolic(i, j, k,
                                        Flux, cmax_m,
                                        U_Array,
                                        gamma,
                                        ID_Dimension);
            Conservative2FluxHyperbolic(i, j, k+1,
                                        Flux, cmax_p,
                                        U_Array,
                                        gamma,
                                        ID_Dimension);
            WaveSpeed_Array(i,j,k,ID_Dimension) = std::max(cmax_m, cmax_p);
#endif
        });
    }
    for (int idim = 0; idim < AMREX_SPACEDIM; idim++){
        max_wave_speed[idim] = WaveSpeed.norminf(idim);
    }
}