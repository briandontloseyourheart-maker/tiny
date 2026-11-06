# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../../amrex


DEBUG        = FALSE
USE_MPI      = TRUE
MPI_THREAD_MULTIPLE = FALSE  # amrex.async_out=1 with more than 64 processes requires MPI_THREAD_MULTIPLE
USE_OMP      = FALSE
USE_CUDA     = FALSE
USE_HIP      = FALSE
USE_SYCL     = FALSE

USE_HDF5     = FALSE

COMP         = gnu
DIM          = 2


include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(USE_HDF5),TRUE)
	# find HDF5 path using "which h5pcc" but without "/bin/h5pcc"
	HDF5_HOME   ?= /usr
    CPP_USE_HDF5 = 1
	INCLUDE_LOCATIONS += $(HDF5_HOME)/include/hdf5/openmpi
	LIBRARY_LOCATIONS += $(HDF5_HOME)/lib/x86_64-linux-gnu/hdf5/openmpi
	LIBRARIES += hdf5
else
    CPP_USE_HDF5 = 0
endif
CXXFLAGS += -DUSE_HDF5=$(CPP_USE_HDF5)

include ../../Source/Make.package
VPATH_LOCATIONS  += ../../Source
INCLUDE_LOCATIONS += ../../Source



Pdirs 	:= Base Boundary LinearSolvers/MLMG
Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
