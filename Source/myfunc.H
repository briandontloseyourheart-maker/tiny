#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "mykernel.H"

using namespace amrex;

void main_main ();

void WriteCheckpoint(const int& step,
                     const amrex::Real& time,
                     const amrex::MultiFab& u,
#if (AMREX_SPACEDIM > 1)
                     const amrex::MultiFab& v,
#endif
#if (AMREX_SPACEDIM > 2)
                     const amrex::MultiFab& w,
#endif
                     const amrex::MultiFab& P,
                     const amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& Vel);

void ReadCheckpoint(const int& restart,
                    amrex::Real& time,
                    amrex::MultiFab& u,
#if (AMREX_SPACEDIM > 1)
                    amrex::MultiFab& v,
#endif
#if (AMREX_SPACEDIM > 2)
                    amrex::MultiFab& w,
#endif
                    amrex::MultiFab& P,
                    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& Vel,
                    amrex::BoxArray& ba,
                    amrex::DistributionMapping& dm);

void advance (amrex::MultiFab& u_old, amrex::MultiFab& u_new, BoundaryCondition const& BoundaryCondition_u,
#if (AMREX_SPACEDIM > 1)
              amrex::MultiFab& v_old, amrex::MultiFab& v_new, BoundaryCondition const& BoundaryCondition_v,
#endif
#if (AMREX_SPACEDIM > 2)
              amrex::MultiFab& w_old, amrex::MultiFab& w_new, BoundaryCondition const& BoundaryCondition_w,
#endif
              amrex::MultiFab& P_old, amrex::MultiFab& P_new, BoundaryCondition const& BoundaryCondition_P,
              amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& Vel_old, amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& Vel_new,
              amrex::Array<amrex::Real,AMREX_SPACEDIM>& Force_Hydrodynamic,
              amrex::Geometry const& geom,
              amrex::Real const& dt,
              amrex::Real const& time,
              amrex::Real const& nu,
              amrex::Real const& tol_rel,
              int max_iter,
              int max_fmg_iter,
              int verbose,
              int bottom_verbose);

void Initial_Condition (amrex::MultiFab& u_new, BoundaryCondition const& BoundaryCondition_u,
#if (AMREX_SPACEDIM > 1)
                        amrex::MultiFab& v_new, BoundaryCondition const& BoundaryCondition_v,
#endif
#if (AMREX_SPACEDIM > 2)
                        amrex::MultiFab& w_new, BoundaryCondition const& BoundaryCondition_w,
#endif
                        amrex::MultiFab& P_new,
                        amrex::Array<amrex::MultiFab,AMREX_SPACEDIM>& Vel_new,
                        amrex::Geometry const& geom);

void Boundary_Condition (amrex::MultiFab& f,
                         BoundaryCondition const& BoundaryCondition_f,
                         amrex::Geometry const& geom);
#endif
