#!/bin/env python

import sys
import re
import numpy as np
import h5py

import virgo.mpi.parallel_hdf5
import virgo.mpi.parallel_sort as ps

import lustre
import command_line_args
import read_vr

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


if __name__ == "__main__":

    # Read command line parameters
    args = command_line_args.get_group_membership_args(comm)

    # Ensure output dir exists
    if comm_rank == 0:
        lustre.ensure_output_dir(args.output_file)
    comm.barrier()

    # Find group number for each particle ID in the VR output
    ids_bound, grnr_bound, ids_unbound, grnr_unbound = read_vr.find_group_membership(args.vr_basename)

    # Determine SWIFT particle types which exist in the snapshot
    ptypes = []
    with h5py.File(args.swift_filename % {"file_nr" : 0}, "r") as infile:
        nr_types = infile["Header"].attrs["NumPartTypes"][0]
        numpart_total = (infile["Header"].attrs["NumPart_Total"].astype(np.int64) +
                         infile["Header"].attrs["NumPart_Total_HighWord"].astype(np.int64) << 32)
        nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]
        for i in range(nr_types):
            if numpart_total[i] > 0:
                ptypes.append("PartType%d" % i)

    # Open the snapshot
    snap_file = virgo.mpi.parallel_hdf5.MultiFile(args.swift_filename,
                                                  file_nr_attr=("Header", "NumFilesPerSnapshot"))

    # Loop over particle types
    create_file = True
    for ptype in ptypes:

        if comm_rank == 0:
            print("Calculating group membership for type ", ptype)
        swift_ids = snap_file.read(("ParticleIDs",), ptype)["ParticleIDs"]

        # Allocate array to store SWIFT particle group membership
        swift_grnr_bound   = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)
        swift_grnr_unbound = np.ndarray(len(swift_ids), dtype=grnr_bound.dtype)

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to VR bound IDs")
        ptr = ps.parallel_match(swift_ids, ids_bound)
        
        if comm_rank == 0:
            print("  Assigning VR bound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_bound[matched] = ps.fetch_elements(grnr_bound, ptr[matched])
        swift_grnr_bound[matched==False] = -1

        if comm_rank == 0:
            print("  Matching SWIFT particle IDs to VR unbound IDs")
        ptr = ps.parallel_match(swift_ids, ids_unbound)
        
        if comm_rank == 0:
            print("  Assigning VR unbound group membership to SWIFT particles")
        matched = ptr >= 0
        swift_grnr_unbound[matched] = ps.fetch_elements(grnr_unbound, ptr[matched])
        swift_grnr_unbound[matched==False] = -1
        swift_grnr_all = np.maximum(swift_grnr_bound, swift_grnr_unbound)

        # Determine if we need to create a new output file set
        if create_file:
            mode="w"
            create_file=False
        else:
            mode="r+"

        # Set up dataset attributes
        unit_attrs = {
            "Conversion factor to CGS (not including cosmological corrections)" : [1.0,],
            "Conversion factor to CGS (including cosmological corrections)" : [1.0,],
            "U_I exponent" : [0.0,],
            "U_L exponent" : [0.0,],
            "U_M exponent" : [0.0,],
            "U_t exponent" : [0.0,],
            "U_T exponent" : [0.0,],
            "a-scale exponent" : [0.0,],
            "h-scale exponent" : [0.0,],
        }
        attrs = {"GroupNr_bound" : {"Description" : "Index of halo in which this particle is a bound member, or -1 if none"},
                 "GroupNr_all" : {"Description" : "Index of halo in which this particle is a member (bound or unbound), or -1 if none"}}
        attrs["GroupNr_bound"].update(unit_attrs)
        attrs["GroupNr_all"].update(unit_attrs)

        # Write these particles out with the same layout as the snapshot
        if comm_rank == 0:
            print("  Writing out VR group membership of SWIFT particles")
        elements_per_file = snap_file.get_elements_per_file("ParticleIDs", group=ptype)
        output = {"GroupNr_bound"   : swift_grnr_bound,
                  "GroupNr_all"     : swift_grnr_all}
        snap_file.write(output, elements_per_file, filenames=args.output_file, mode=mode, group=ptype, attrs=attrs)

    comm.barrier()
    if comm_rank == 0:
        print("Done.")

