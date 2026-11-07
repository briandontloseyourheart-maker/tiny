#ifndef MY_FUNC_H_
#define MY_FUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "mykernel.H"

using namespace amrex;

void main_main ();

void WriteCheckpoint(const int& step,
                     const amrex::Real& time,
                     const amrex::MultiFab& U);

void ReadCheckpoint(const int& restart,
                    amrex::Real& time,
                    amrex::MultiFab& U,
                    amrex::BoxArray& ba,
                    amrex::DistributionMapping& dm,
                    int Nghost);

void advance (amrex::MultiFab& U_old, amrex::MultiFab& U_new,
              amrex::Array<BoundaryCondition,NSTATE> const& BoundaryCondition_V,
              amrex::Array<amrex::Real,AMREX_SPACEDIM>& Force_Hydrodynamic,
              amrex::Geometry const& geom,
              amrex::Real const& dt,
              amrex::Real const& time,
              amrex::Real const& gamma,
              amrex::Real const& nu,
              ImmersedBoundary const& immersed_boundary,
              amrex::Real const& tol_rel,
              int max_iter,
              int max_fmg_iter,
              int verbose,
              int bottom_verbose);

void Initial_Condition (amrex::MultiFab& U_new,
                        amrex::Geometry const& geom,
                        InitialCondition const& initial_condition,
                        amrex::Real const& gamma);

void Boundary_Condition (amrex::MultiFab& U,
                         amrex::Array<BoundaryCondition,NSTATE> const& BoundaryCondition_V,
                         amrex::Geometry const& geom,
                         amrex::Real const& time,
                         amrex::Real const& gamma);

void Max_WaveSpeed (amrex::Array<amrex::Real,AMREX_SPACEDIM>& max_wave_speed,
                    amrex::MultiFab& U,
                    amrex::Array<BoundaryCondition,NSTATE> const& BoundaryCondition_V,
                    amrex::Geometry const& geom,
                    amrex::Real const& time,
                    amrex::Real const& gamma);
#endif
