#ifndef MY_STRUCT_H_
#define MY_STRUCT_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_Random.H>
#include <AMReX_Math.H>

struct InitialCondition
{
    amrex::Parser u_parser;
    amrex::ParserExecutor<AMREX_SPACEDIM> u;
    amrex::Real u_noise = 0.0;
#if (AMREX_SPACEDIM > 1)
    amrex::Parser v_parser;
    amrex::ParserExecutor<AMREX_SPACEDIM> v;
    amrex::Real v_noise = 0.0;
#endif
#if (AMREX_SPACEDIM > 2)
    amrex::Parser w_parser;
    amrex::ParserExecutor<AMREX_SPACEDIM> w;
    amrex::Real w_noise = 0.0;
#endif
    amrex::Parser P_parser;
    amrex::ParserExecutor<AMREX_SPACEDIM> P;
    amrex::Real P_noise = 0.0;

    int noise_seed = 12345;

    void Initialize ()
    {
        amrex::ParmParse pp("IC");

        std::string u_expr = "0.0001";
        pp.query("u", u_expr);
        u_parser.define(u_expr);
        u_parser.registerVariables({AMREX_D_DECL("x", "y", "z")});
        u = u_parser.compile<AMREX_SPACEDIM>();

        pp.query("u_noise", u_noise);
        if (u_noise < 0.0){
            amrex::Abort("IC.u_noise must be >= 0.0");
        }
#if (AMREX_SPACEDIM > 1)
        std::string v_expr = "0.0001";
        pp.query("v", v_expr);
        v_parser.define(v_expr);
        v_parser.registerVariables({AMREX_D_DECL("x", "y", "z")});
        v = v_parser.compile<AMREX_SPACEDIM>();

        pp.query("v_noise", v_noise);
        if (v_noise < 0.0){
            amrex::Abort("IC.v_noise must be >= 0.0");
        }
#endif
#if (AMREX_SPACEDIM > 2)
        std::string w_expr = "0.001";
        pp.query("w", w_expr);
        w_parser.define(w_expr);
        w_parser.registerVariables({AMREX_D_DECL("x", "y", "z")});
        w = w_parser.compile<AMREX_SPACEDIM>();

        pp.query("w_noise", w_noise);
        if (w_noise < 0.0){
            amrex::Abort("IC.w_noise must be >= 0.0");
        }
#endif
        std::string P_expr = "0.0";
        pp.query("P", P_expr);
        P_parser.define(P_expr);
        P_parser.registerVariables({AMREX_D_DECL("x", "y", "z")});
        P = P_parser.compile<AMREX_SPACEDIM>();

        pp.query("P_noise", P_noise);
        if (P_noise < 0.0){
            amrex::Abort("IC.P_noise must be >= 0.0");
        }

        pp.query("noise_seed", noise_seed);
        if (noise_seed < 0){
            amrex::Abort("IC.noise_seed must be >= 0");
        }
        // Make the optional random perturbation reproducible while giving
        // different MPI ranks independent CPU/GPU random streams.
        const amrex::ULong base_seed = static_cast<amrex::ULong>(noise_seed);
        const amrex::ULong rank_seed = base_seed + 104729ULL*static_cast<amrex::ULong>(amrex::ParallelDescriptor::MyProc());
        amrex::ResetRandomSeed(rank_seed, rank_seed);
    }
};


struct ImmersedBoundary
{
    amrex::Parser x_parser;
    amrex::ParserExecutor<1> x;
#if (AMREX_SPACEDIM > 1)
    amrex::Parser y_parser;
    amrex::ParserExecutor<1> y;
#endif
#if (AMREX_SPACEDIM > 2)
    amrex::Parser z_parser;
    amrex::ParserExecutor<1> z;
#endif
    amrex::Real radius  = 0.5;
    amrex::Real epsilon = 0.01;

    void Initialize ()
    {
        amrex::ParmParse pp("IB");

        std::string x_expr = "0.0";
        pp.query("x", x_expr);
        x_parser.define(x_expr);
        x_parser.registerVariables({"t"});
        x = x_parser.compile<1>();
#if (AMREX_SPACEDIM > 1)
        std::string y_expr = "0.0";
        pp.query("y", y_expr);
        y_parser.define(y_expr);
        y_parser.registerVariables({"t"});
        y = y_parser.compile<1>();
#endif
#if (AMREX_SPACEDIM > 2)
        std::string z_expr = "0.0";
        pp.query("z", z_expr);
        z_parser.define(z_expr);
        z_parser.registerVariables({"t"});
        z = z_parser.compile<1>();
#endif
        pp.query("radius", radius);
        if (radius <= 0.0){
            amrex::Abort("IB.radius must be > 0.0");
        }
        pp.query("epsilon", epsilon);
        if (epsilon <= 0.0){
            amrex::Abort("IB.epsilon must be > 0.0");
        }
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Array<amrex::Real,AMREX_SPACEDIM> Velocity (amrex::Real const& time, amrex::Real const& dt) const
    {
        amrex::Real X_pls, X_mis;

        X_pls = x(time + dt);
        X_mis = x(time - dt);
        amrex::Real u = (X_pls - X_mis)/(2.0*dt);
#if (AMREX_SPACEDIM > 1)
        X_pls = y(time + dt);
        X_mis = y(time - dt);
        amrex::Real v = (X_pls - X_mis)/(2.0*dt);
#endif
#if (AMREX_SPACEDIM > 2)
        X_pls = z(time + dt);
        X_mis = z(time - dt);
        amrex::Real w = (X_pls - X_mis)/(2.0*dt);
#endif

        return {AMREX_D_DECL(u, v, w)};
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Array<amrex::Real,AMREX_SPACEDIM> Acceleration (amrex::Real const& time, amrex::Real const& dt) const
    {
        amrex::Real X_pls, X_ctr, X_mis;

        X_pls = x(time + dt);
        X_ctr = x(time);
        X_mis = x(time - dt);
        amrex::Real ax = (X_pls - 2.0*X_ctr + X_mis)/(dt*dt);
#if (AMREX_SPACEDIM > 1)
        X_pls = y(time + dt);
        X_ctr = y(time);
        X_mis = y(time - dt);
        amrex::Real ay = (X_pls - 2.0*X_ctr + X_mis)/(dt*dt);
#endif
#if (AMREX_SPACEDIM > 2)
        X_pls = z(time + dt);
        X_ctr = z(time);
        X_mis = z(time - dt);
        amrex::Real az = (X_pls - 2.0*X_ctr + X_mis)/(dt*dt);
#endif

        return {AMREX_D_DECL(ax, ay, az)};
    }
};


enum class BoundaryConditionType : int {
    Periodic,
    Dirichlet,
    Neumann,
    Robin,
    User
};

struct BoundaryCondition
{
    amrex::Array<BoundaryConditionType,AMREX_SPACEDIM> lo;
    amrex::Array<BoundaryConditionType,AMREX_SPACEDIM> hi;

    amrex::Array<amrex::Parser,AMREX_SPACEDIM> lo_Dirichlet_parser;
    amrex::Array<amrex::Parser,AMREX_SPACEDIM> hi_Dirichlet_parser;
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> lo_Dirichlet;
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> hi_Dirichlet;

    amrex::Array<amrex::Parser,AMREX_SPACEDIM> lo_Neumann_parser;
    amrex::Array<amrex::Parser,AMREX_SPACEDIM> hi_Neumann_parser;
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> lo_Neumann;
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> hi_Neumann;

    amrex::Array<amrex::Parser,AMREX_SPACEDIM> lo_Function_parser;
    amrex::Array<amrex::Parser,AMREX_SPACEDIM> hi_Function_parser;
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> lo_Function;
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM+1>,AMREX_SPACEDIM> hi_Function;

    void Initialize (std::string const& BC_Variable,
                     amrex::Array<int,AMREX_SPACEDIM> const& is_periodic)
    {
        amrex::ParmParse pp(BC_Variable);

        amrex::Vector<std::string> lo_Dirichlet_expr(AMREX_SPACEDIM, "0.0");
        amrex::Vector<std::string> hi_Dirichlet_expr(AMREX_SPACEDIM, "0.0");
        pp.queryarr("lo_Dirichlet", lo_Dirichlet_expr);
        pp.queryarr("hi_Dirichlet", hi_Dirichlet_expr);

        amrex::Vector<std::string> lo_Neumann_expr(AMREX_SPACEDIM, "0.0");
        amrex::Vector<std::string> hi_Neumann_expr(AMREX_SPACEDIM, "0.0");
        pp.queryarr("lo_Neumann", lo_Neumann_expr);
        pp.queryarr("hi_Neumann", hi_Neumann_expr);

        amrex::Vector<std::string> lo_Function_expr(AMREX_SPACEDIM, "0.0");
        amrex::Vector<std::string> hi_Function_expr(AMREX_SPACEDIM, "0.0");
        pp.queryarr("lo_Function", lo_Function_expr);
        pp.queryarr("hi_Function", hi_Function_expr);

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            lo_Dirichlet_parser[idim].define(lo_Dirichlet_expr[idim]);
            lo_Dirichlet_parser[idim].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            lo_Dirichlet[idim] = lo_Dirichlet_parser[idim].compile<AMREX_SPACEDIM+1>();

            hi_Dirichlet_parser[idim].define(hi_Dirichlet_expr[idim]);
            hi_Dirichlet_parser[idim].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            hi_Dirichlet[idim] = hi_Dirichlet_parser[idim].compile<AMREX_SPACEDIM+1>();

            lo_Neumann_parser[idim].define(lo_Neumann_expr[idim]);
            lo_Neumann_parser[idim].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            lo_Neumann[idim] = lo_Neumann_parser[idim].compile<AMREX_SPACEDIM+1>();

            hi_Neumann_parser[idim].define(hi_Neumann_expr[idim]);
            hi_Neumann_parser[idim].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            hi_Neumann[idim] = hi_Neumann_parser[idim].compile<AMREX_SPACEDIM+1>();

            lo_Function_parser[idim].define(lo_Function_expr[idim]);
            lo_Function_parser[idim].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            lo_Function[idim] = lo_Function_parser[idim].compile<AMREX_SPACEDIM+1>();

            hi_Function_parser[idim].define(hi_Function_expr[idim]);
            hi_Function_parser[idim].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            hi_Function[idim] = hi_Function_parser[idim].compile<AMREX_SPACEDIM+1>();

            if (is_periodic[idim] == 1){
                lo[idim] = BoundaryConditionType::Periodic;
                hi[idim] = BoundaryConditionType::Periodic;
            }
            else{
                if (lo_Dirichlet_expr[idim] != "0.0" && lo_Neumann_expr[idim] == "0.0"){
                    lo[idim] = BoundaryConditionType::Dirichlet;
                }
                else if (lo_Dirichlet_expr[idim] == "0.0" && lo_Neumann_expr[idim] != "0.0"){
                    lo[idim] = BoundaryConditionType::Neumann;
                }
                else{
                    lo[idim] = BoundaryConditionType::Robin;
                }

                if (hi_Dirichlet_expr[idim] != "0.0" && hi_Neumann_expr[idim] == "0.0"){
                    hi[idim] = BoundaryConditionType::Dirichlet;
                }
                else if (hi_Dirichlet_expr[idim] == "0.0" && hi_Neumann_expr[idim] != "0.0"){
                    hi[idim] = BoundaryConditionType::Neumann;
                }
                else{
                    hi[idim] = BoundaryConditionType::Robin;
                }
            }
        }
    }
};
#endif
