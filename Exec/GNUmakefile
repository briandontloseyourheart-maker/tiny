# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex

# find HDF5 path using "which h5pcc" but without "/bin/h5pcc"
HDF5_HOME    =/usr


DEBUG        = FALSE
USE_MPI      = TRUE
MPI_THREAD_MULTIPLE = FALSE  # amrex.async_out=1 with more than 64 processes requires MPI_THREAD_MULTIPLE
USE_OMP      = FALSE
USE_CUDA     = FALSE
USE_HIP      = FALSE
USE_SYCL     = FALSE

USE_HDF5     = TRUE

COMP         = gnu
DIM          = 2

INCLUDE_LOCATIONS += /usr/include/hdf5/openmpi
LIBRARY_LOCATIONS += /usr/lib/x86_64-linux-gnu/hdf5/openmpi


include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ../Source/Make.package
VPATH_LOCATIONS  += ../Source
INCLUDE_LOCATIONS += ../Source



Pdirs 	:= Base Boundary LinearSolvers/MLMG
Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
