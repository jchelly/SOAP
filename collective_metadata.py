#!/bin/env python

import h5py

def create_file(filename, comm):
    """
    Create a HDF5 file with the mpio driver and collective metadata enabled.
    Returns a writable h5py.File
    """
    
    # Open file with low level interface, with collective metadata mode
    mpi_info = MPI.Info.Create()
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_fapl_mpio(comm, mpi_info)
    fapl.set_coll_metadata_write(True)
    fapl.set_all_coll_metadata_ops(True)
    f = h5py.File(h5py.h5f.create(filename.encode(), fcpl=fcpl, fapl=fapl))
 
    return f


def open_file(filename, comm):
    """
    Open a HDF5 file with the mpio driver and collective metadata enabled.
    Returns a read only h5py.File
    """
    
    # Open file with low level interface, with collective metadata mode
    mpi_info = MPI.Info.Create()
    fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    fapl.set_fapl_mpio(comm, mpi_info)
    fapl.set_coll_metadata_write(True)
    fapl.set_all_coll_metadata_ops(True)
    f = h5py.File(h5py.h5f.open(filename.encode(), flags=h5py.h5f.ACC_RDONLY, fapl=fapl))
 
    return f
