#include <AMReX_Gpu.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <filesystem>

#include "myfunc.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{
    // What time is it now?  We'll use this to compute total run time.
    auto strt_time = ParallelDescriptor::second();

    // AMREX_SPACEDIM: number of dimensions
    Real AMREX_D_DECL(lo_x, lo_y, lo_z);
    Real AMREX_D_DECL(hi_x, hi_y, hi_z);

    int AMREX_D_DECL(n_cell_x, n_cell_y, n_cell_z);
    int max_grid_size;

    Array<int,AMREX_SPACEDIM> is_periodic;
    BoundaryCondition AMREX_D_DECL(BoundaryCondition_u, BoundaryCondition_v, BoundaryCondition_w), BoundaryCondition_P;

    Real Final_Time, dt, CFL;
    int Fixed_TimeStep;

    Real Re, nu;

    Real tol_rel;
    int max_iter, max_fmg_iter, verbose, bottom_verbose;

    int nsteps, plot_int, plotForce_int;

    // IO_InfoSteps: interval for Steps/Time/CPU Time information
    // IO_CFLSteps : interval for an additional CFL summary
    int IO_InfoSteps;
    int IO_CFLSteps;

    // how often to write a checkpoint file
    int chk_int;

    // restart = -1 --> start new simulation
    // restart >  0 --> restart from step=restart
	
    int restart;

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        ParmParse pp;
        Vector<int> Vector_Int(AMREX_SPACEDIM);
        Vector<Real> Vector_Real(AMREX_SPACEDIM);

        AMREX_D_TERM(
        pp.get("lo_x",lo_x);,
        pp.get("lo_y",lo_y);,
        pp.get("lo_z",lo_z));

        AMREX_D_TERM(
        pp.get("hi_x",hi_x);,
        pp.get("hi_y",hi_y);,
        pp.get("hi_z",hi_z));

        AMREX_D_TERM(
        pp.get("n_cell_x",n_cell_x);,
        pp.get("n_cell_y",n_cell_y);,
        pp.get("n_cell_z",n_cell_z));

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        pp.getarr("is_periodic",Vector_Int);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            is_periodic[idim] = Vector_Int[idim];
        }

        pp.getarr("lo_Dirichlet_u",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_u.lo_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Dirichlet_u",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_u.hi_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Neumann_u"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_u.lo_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Neumann_u"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_u.hi_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Function_u" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_u.lo_Function[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Function_u" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_u.hi_Function[idim] = Vector_Real[idim];
        }
        BoundaryCondition_u.Initialize(is_periodic);
#if (AMREX_SPACEDIM > 1)
        pp.getarr("lo_Dirichlet_v",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_v.lo_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Dirichlet_v",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_v.hi_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Neumann_v"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_v.lo_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Neumann_v"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_v.hi_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Function_v" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_v.lo_Function[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Function_v" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_v.hi_Function[idim] = Vector_Real[idim];
        }
        BoundaryCondition_v.Initialize(is_periodic);
#endif
#if (AMREX_SPACEDIM > 2)
        pp.getarr("lo_Dirichlet_w",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_w.lo_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Dirichlet_w",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_w.hi_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Neumann_w"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_w.lo_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Neumann_w"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_w.hi_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Function_w" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_w.lo_Function[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Function_w" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_w.hi_Function[idim] = Vector_Real[idim];
        }
        BoundaryCondition_w.Initialize(is_periodic);
#endif
        pp.getarr("lo_Dirichlet_P",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_P.lo_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Dirichlet_P",Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_P.hi_Dirichlet[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Neumann_P"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_P.lo_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Neumann_P"  ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_P.hi_Neumann[idim] = Vector_Real[idim];
        }
        pp.getarr("lo_Function_P" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_P.lo_Function[idim] = Vector_Real[idim];
        }
        pp.getarr("hi_Function_P" ,Vector_Real);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            BoundaryCondition_P.hi_Function[idim] = Vector_Real[idim];
        }
        BoundaryCondition_P.Initialize(is_periodic);

        pp.get("Final_Time",Final_Time);
        pp.get("Fixed_TimeStep",Fixed_TimeStep);
        if (Fixed_TimeStep){
            pp.get("dt",dt);
        }
        else{
            pp.get("CFL",CFL);
        }

        pp.get("Re",Re);
        nu = 1.0/Re;

        tol_rel = 1.0e-10;
        pp.query("tol_rel",tol_rel);

        max_iter = 100;
        pp.query("max_iter",max_iter);

        max_fmg_iter = 0;
        pp.query("max_fmg_iter",max_fmg_iter);

        verbose = 0;
        pp.query("verbose",verbose);

        bottom_verbose = 0;
        pp.query("bottom_verbose",bottom_verbose);

        // Default plot_int to -1, allow us to set it to something else in the inputs file
        //  If plot_int < 0 then no plot files will be written
        plot_int = -1;
        pp.query("plot_int",plot_int);

        plotForce_int = -1;
        pp.query("plotForce_int",plotForce_int);

        // screen-output controls.
        // Set <= 0 to disable the corresponding periodic output.
        IO_InfoSteps = 200;
        IO_CFLSteps  = 100;
        pp.query("IO_InfoSteps", IO_InfoSteps);
        pp.query("IO_CFLSteps",  IO_CFLSteps);

        // Default chk_int to -1, allow us to set it to something else in the inputs file
        //  If chk_int < 0 then no chk files will be written
        chk_int = -1;
        pp.query("chk_int",chk_int);

        // restart = -1 --> start new simulation
        // restart >  0 --> restart from step=restart
        restart = -1;
        pp.query("restart",restart);

        // Default nsteps to 10, allow us to set it to something else in the inputs file
        nsteps = 10;
        pp.query("nsteps",nsteps);
    }

    // make Geometry
    Geometry geom;
    {
        IntVect dom_lo(AMREX_D_DECL(         0,          0,          0));
        IntVect dom_hi(AMREX_D_DECL(n_cell_x-1, n_cell_y-1, n_cell_z-1));
        Box domain(dom_lo, dom_hi);
        // This defines the physical box, [-1,1] in each direction.
        RealBox real_box({AMREX_D_DECL(lo_x, lo_y, lo_z)},
                         {AMREX_D_DECL(hi_x, hi_y, hi_z)});
        // This defines a Geometry object
        geom.define(domain,real_box,CoordSys::cartesian,is_periodic);
    }

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // make BoxArray and DistributionMapping
    BoxArray ba;
    DistributionMapping dm;

    // we allocate two phi multifabs; one will store the old state, the other the new.
    MultiFab u_old, u_new;
#if (AMREX_SPACEDIM > 1)
    MultiFab v_old, v_new;
#endif
#if (AMREX_SPACEDIM > 2)
    MultiFab w_old, w_new;
#endif
    MultiFab P_old, P_new;
    Array<MultiFab,AMREX_SPACEDIM> Vel_old, Vel_new;
    Array<Real,AMREX_SPACEDIM> Force_Hydrodynamic = {AMREX_D_DECL(0.0, 0.0, 0.0)};
    // ========================================

    Real time;
    int start_step;

    // start a new simulation
    if (restart <= 0){
        // Initialize the boxarray "ba" from the single box "bx"
        ba.define(geom.Domain());
        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

        // How Boxes are distrubuted among MPI processes
        dm.define(ba);

        u_old.define(ba, dm, 1, Nghost);
        u_new.define(ba, dm, 1, Nghost);
#if (AMREX_SPACEDIM > 1)
        v_old.define(ba, dm, 1, Nghost);
        v_new.define(ba, dm, 1, Nghost);
#endif
#if (AMREX_SPACEDIM > 2)
         w_old.define(ba, dm, 1, Nghost);
         w_new.define(ba, dm, 1, Nghost);
#endif
        P_old.define(ba, dm, 1, Nghost);
        P_new.define(ba, dm, 1, Nghost);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray ba_face = ba;
            ba_face.surroundingNodes(idim);
            Vel_old[idim].define(ba_face, dm, 1, 0);
            Vel_new[idim].define(ba_face, dm, 1, 0);
        }

        // time = starting time in the simulation
        time = 0.0;
        start_step = 1;
        Initial_Condition(u_new, BoundaryCondition_u,
#if (AMREX_SPACEDIM > 1)
                          v_new, BoundaryCondition_v,
#endif
#if (AMREX_SPACEDIM > 2)
                          w_new, BoundaryCondition_w,
#endif
                          P_new, Vel_new,
                          geom);

        // Write a plotfile of the initial data if plot_int > 0 (plot_int was defined in the inputs file)
        if (plot_int > 0)
        {
            MultiFab Output(ba, dm, AMREX_SPACEDIM+1, 0);
            AMREX_D_TERM(
            MultiFab::Copy(Output, u_new, 0,              0, 1, 0);,
            MultiFab::Copy(Output, v_new, 0,              1, 1, 0);,
            MultiFab::Copy(Output, w_new, 0,              2, 1, 0));
            MultiFab::Copy(Output, P_new, 0, AMREX_SPACEDIM, 1, 0);

            int n = 0;
            const std::string& pltfile = amrex::Concatenate("./plot/plt",n,5);
            WriteSingleLevelPlotfile(pltfile, Output, {AMREX_D_DECL("u", "v", "w"), "P"}, geom, time, 0);
#if (USE_HDF5)
            std::filesystem::create_directories("./plot_HDF5");
			const std::string& pltfile_HDF5 = amrex::Concatenate("./plot_HDF5/plt",n,5);
			WriteSingleLevelPlotfileHDF5(pltfile_HDF5, Output, {AMREX_D_DECL("u", "v", "w"), "P"}, geom, time, 0, "None@0");
#endif
        }
        if (plotForce_int > 0)
        {
            // ---------------------------
            // File I/O setup
            // ---------------------------
            const std::string fname = "./Force_Hydrodynamic.csv";

            std::fstream file;
            file.open(fname, std::ofstream::out | std::ofstream::trunc);
            file.precision(18);

            // ---------------------------
            // Write header + row
            // ---------------------------
            if (amrex::ParallelDescriptor::MyProc() == 0)
            {
                file << "Time, Force_x, Force_y, Force_z \n";

                Real Force_x = Force_Hydrodynamic[0];
#if (AMREX_SPACEDIM > 1)
                Real Force_y = Force_Hydrodynamic[1];
#else
                Real Force_y = 0.0;
#endif
#if (AMREX_SPACEDIM > 2)
                Real Force_z = Force_Hydrodynamic[2];
#else
                Real Force_z = 0.0;
#endif
                file << time << "," << Force_x << "," << Force_y << "," << Force_z << "\n";
            }
            file.close();
        }
    }
    else{
        // restart a simulation
        // read in the time and BoxArray, then create a DistributionMapping
        // Define phi_old and fill it with data from the checkpoint file
        ReadCheckpoint(restart, time,
                       u_new,
#if (AMREX_SPACEDIM > 1)
                       v_new,
#endif
#if (AMREX_SPACEDIM > 2)
                       w_new,
#endif
                       P_new, Vel_new,
                       ba, dm);

        // set start_step to restart+1
        start_step = restart + 1;

        u_old.define(ba, dm, 1, Nghost);
#if (AMREX_SPACEDIM > 1)
        v_old.define(ba, dm, 1, Nghost);
#endif
#if (AMREX_SPACEDIM > 2)
        w_old.define(ba, dm, 1, Nghost);
#endif
        P_old.define(ba, dm, 1, Nghost);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            BoxArray ba_face = ba;
            ba_face.surroundingNodes(idim);
            Vel_old[idim].define(ba_face, dm, 1, 0);
        }
    }

    // -------------------------------------------------------------------------
    // Print a run summary
    // -------------------------------------------------------------------------
    amrex::Print() << "============================================================\n"
                   << "        Equation Type: Incompressible Navier-Stokes FSI\n"
                   << "         Spatial Dim.: " << AMREX_SPACEDIM << "\n"
                   << "      Num. Processes: " << ParallelDescriptor::NProcs() << "\n"
                   << "        No. of Steps: " << nsteps << "\n"
                   << " Checkpoints (steps): ";
    if (chk_int > 0) {
        amrex::Print() << chk_int << "\n";
    } else {
        amrex::Print() << "Disabled\n";
    }
    amrex::Print() << "      IO InfoSteps: ";
    if (IO_InfoSteps > 0) {
        amrex::Print() << IO_InfoSteps << "\n";
    } else {
        amrex::Print() << "Disabled\n";
    }
    amrex::Print() << "       IO CFLSteps: ";
    if (IO_CFLSteps > 0) {
        amrex::Print() << IO_CFLSteps << "\n";
    } else {
        amrex::Print() << "Disabled\n";
    }
	
    if (restart <= 0) {
        amrex::Print() << "Initial Conditions:\n"
                       << "  - Field u: 0.0001\n";
#if (AMREX_SPACEDIM > 1)
        amrex::Print() << "  - Field v: 0.0001\n";
#endif
#if (AMREX_SPACEDIM > 2)
        amrex::Print() << "  - Field w: 0.001\n";
#endif
        amrex::Print() << "  - Field P: 0.0\n";
    } else {
        amrex::Print() << "Initial Conditions:\n"
                       << "  - Restart from checkpoint: "
                       << amrex::Concatenate("chk", restart, 5) << "\n"
                       << "  - Restart time: " << time << "\n";
    }
    amrex::Print() << "============================================================\n";

    // Timer used for Nektar++-style interval timing.  ParallelDescriptor::second()
    // measures wall-clock time; the reported value below is the maximum over MPI ranks.
    auto screen_time_start = ParallelDescriptor::second();
    const int IOProc = ParallelDescriptor::IOProcessorNumber();

    // Time advancement
    for (int n = start_step; n <= nsteps && time < Final_Time; ++n)
    {
        // Determine time step
        if (!Fixed_TimeStep){
            GpuArray<Real,AMREX_SPACEDIM> dX = geom.CellSizeArray();
            Real dt_Adv = CFL/std::max({AMREX_D_DECL(Vel_new[0].norminf()/dX[0], Vel_new[1].norminf()/dX[1], Vel_new[2].norminf()/dX[2])});
            Real dt_Dif = CFL/nu/(AMREX_D_TERM(2.0/dX[0]/dX[0], + 2.0/dX[1]/dX[1], + 2.0/dX[2]/dX[2]));
            dt = std::min(dt_Dif, dt_Adv);
        }
        if (time + dt > Final_Time){
            dt = Final_Time - time;
        }
        GpuArray<Real,AMREX_SPACEDIM> dX = geom.CellSizeArray();
        Real CFL_Adv = dt*std::max({AMREX_D_DECL(Vel_new[0].norminf()/dX[0], Vel_new[1].norminf()/dX[1], Vel_new[2].norminf()/dX[2])});
        Real CFL_Dif = nu*dt*(AMREX_D_TERM(2.0/dX[0]/dX[0], + 2.0/dX[1]/dX[1], + 2.0/dX[2]/dX[2]));

        AMREX_D_TERM(
        MultiFab::Copy(u_old, u_new, 0, 0, 1, 0);,
        MultiFab::Copy(v_old, v_new, 0, 0, 1, 0);,
        MultiFab::Copy(w_old, w_new, 0, 0, 1, 0));
        MultiFab::Copy(P_old, P_new, 0, 0, 1, 0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim){
            MultiFab::Copy(Vel_old[idim], Vel_new[idim], 0, 0, 1, 0);
        }

        // new_phi = old_phi + dt * (something)
        advance(u_old, u_new, BoundaryCondition_u,
#if (AMREX_SPACEDIM > 1)
                v_old, v_new, BoundaryCondition_v,
#endif
#if (AMREX_SPACEDIM > 2)
                w_old, w_new, BoundaryCondition_w,
#endif
                P_old, P_new, BoundaryCondition_P,
                Vel_old, Vel_new, Force_Hydrodynamic,
                geom, dt, time,
                nu,
                tol_rel, max_iter, max_fmg_iter, verbose, bottom_verbose);
        time = time + dt;

        // The per-step progress output.
        // amrex::Print() << "Advanced step " << n
        //               << " at Time = " << time
        //               << " with dt = " << dt
        //               << " CFL_Adv = " << CFL_Adv
        //               << " CFL_Dif = " << CFL_Dif << "\n";

        // CFL summary per IO_CFLSteps.
     
        if (IO_CFLSteps > 0 && n % IO_CFLSteps == 0)
        {
            amrex::Print()  << "Advanced step " << n
							<< " at Time = " << time
							<< " CFL: Adv = " << CFL_Adv
                           << "    Dif = " << CFL_Dif
                           << "    dt = " << dt << "\n";
        }

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && n%plot_int == 0)
        {
            MultiFab Output(ba, dm, AMREX_SPACEDIM+1, 0);
            AMREX_D_TERM(
            MultiFab::Copy(Output, u_new, 0,              0, 1, 0);,
            MultiFab::Copy(Output, v_new, 0,              1, 1, 0);,
            MultiFab::Copy(Output, w_new, 0,              2, 1, 0));
            MultiFab::Copy(Output, P_new, 0, AMREX_SPACEDIM, 1, 0);

            const std::string& pltfile = amrex::Concatenate("./plot/plt",n,5);
            WriteSingleLevelPlotfile(pltfile, Output, {AMREX_D_DECL("u", "v", "w"), "P"}, geom, time, n);
#if (USE_HDF5)
			std::filesystem::create_directories("./plot_HDF5");
            const std::string& pltfile_HDF5 = amrex::Concatenate("./plot_HDF5/plt",n,5);
			WriteSingleLevelPlotfileHDF5(pltfile_HDF5, Output, {AMREX_D_DECL("u", "v", "w"), "P"}, geom, time, n, "None@0");
#endif
        }
        if (plotForce_int > 0 && n%plotForce_int == 0)
        {
            // ---------------------------
            // File I/O setup
            // ---------------------------
            std::fstream file;
			if (restart <= 0){
				const std::string fname = "./Force_Hydrodynamic.csv";
				file.open(fname, std::ios::app);
			}
			else{
				std::filesystem::create_directories("./");
				const std::string& checkpointname = Concatenate("chk",restart,5);
				const std::string fname = "./Force_Hydrodynamic_" + checkpointname + ".csv";
				if (n == restart + 1){
					file.open(fname, std::ofstream::out | std::ofstream::trunc);
				}
				else{
					file.open(fname, std::ios::app);
				}
			}
            file.precision(18);

            // ---------------------------
            // Write header + row
            // ---------------------------
            if (amrex::ParallelDescriptor::MyProc() == 0)
            {
                Real Force_x = Force_Hydrodynamic[0];
#if (AMREX_SPACEDIM > 1)
                Real Force_y = Force_Hydrodynamic[1];
#else
                Real Force_y = 0.0;
#endif
#if (AMREX_SPACEDIM > 2)
                Real Force_z = Force_Hydrodynamic[2];
#else
                Real Force_z = 0.0;
#endif
                file << time << "," << Force_x << "," << Force_y << "," << Force_z << "\n";
            }
            file.close();
        }
        // Write a checkpoint file of the current data (chk_int was defined in the inputs file)
        if (chk_int > 0 && n%chk_int == 0)
        {
            // create a checkpoint directory
            // write out time and BoxArray to a Header file
            // write out multifab data
            WriteCheckpoint(n, time,
                            u_new,
#if (AMREX_SPACEDIM > 1)
                            v_new,
#endif
#if (AMREX_SPACEDIM > 2)
                            w_new,
#endif
                            P_new, Vel_new);
        }

        // Periodically report elapsed wall-clock time for the last IO_InfoSteps steps.
        if (IO_InfoSteps > 0 && n % IO_InfoSteps == 0)
        {
            Real interval_cpu_time = ParallelDescriptor::second() - screen_time_start;
            ParallelDescriptor::ReduceRealMax(interval_cpu_time, IOProc);

            amrex::Print() << "Steps: " << n
                           << "    Time: " << time
                           << "    CPU Time: " << interval_cpu_time << " s\n";

            // Start timing the next IO_InfoSteps-step block.
            screen_time_start = ParallelDescriptor::second();
        }
    }

    // Call the timer again and compute the maximum difference between the start time and stop time
    //   over all processors
    auto stop_time = ParallelDescriptor::second() - strt_time;
    ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

    // Tell the I/O Processor to write out the "run time"
    amrex::Print() << "Run time = " << stop_time << std::endl;
}
