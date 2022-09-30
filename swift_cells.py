#!/bin/env python

import collections
import math

import numpy as np
import h5py
import time
from mpi4py import MPI
import unyt
import scipy.spatial

import swift_units
import shared_array

# HDF5 chunk cache parameters:
# SWIFT writes datasets with large chunks so the default 1Mb may be too small
# (e.g. one uncompressed chunk of positions is ~24Mb in FLAMINGO 2.8Gpc run)
rdcc_nbytes = 250 * 1024 * 1024

# Type to store information about a SWIFT cell for one particle type
swift_cell_t = np.dtype(
    [
        ("centre", np.float64, 3),  # coordinates of cell centre
        ("count", np.int64),  # number of particles in the cell
        ("offset", np.int64),  # offset to first particle
        ("file", np.int32),  # file containing this cell
        ("order", np.int32),  # ordering of the cells in the snapshot file(s)
    ]
)


def identify_datasets(filename, nr_files, ptypes, registry):
    """
    Find units, data type and shape for datasets in snapshot-like files.
    Returns a dict with one entry per particle type. Dict keys are the
    property names and values are (shape, dtype, units) tuples.
    """
    metadata = {ptype: {} for ptype in ptypes}

    # Make a dict of flags of which particle types we still need to find
    to_find = {ptype: True for ptype in ptypes}

    # Scan snapshot files to find shape, type and units for each quantity
    for file_nr in range(nr_files):
        infile = h5py.File(filename % {"file_nr": file_nr}, "r")
        nr_left = 0
        for ptype in ptypes:
            if to_find[ptype]:
                group_name = ptype
                if group_name in infile:
                    for name in infile[group_name]:
                        dset = infile[group_name][name]
                        if "a-scale exponent" in dset.attrs:
                            units = swift_units.units_from_attributes(
                                dict(dset.attrs), registry
                            )
                            dtype = dset.dtype
                            shape = dset.shape[1:]
                            metadata[ptype][name] = (units, dtype, shape)
                    to_find[ptype] = False
                else:
                    nr_left += 1
        infile.close()
        if nr_left == 0:
            break

    return metadata


class SWIFTCellGrid:
    def get_unit(self, name):
        return unyt.Unit(name, registry=self.snap_unit_registry)

    def __init__(
        self,
        snap_filename,
        extra_filename=None,
        snap_filename_ref=None,
        extra_filename_ref=None,
    ):

        self.snap_filename = snap_filename

        # Option format string to generate name of file(s) with extra datasets
        self.extra_filename = extra_filename

        # Open the input file
        with h5py.File(snap_filename % {"file_nr": 0}, "r") as infile:

            # Get the snapshot unit system
            self.snap_unit_registry = swift_units.unit_registry_from_snapshot(infile)
            self.a_unit = self.get_unit("a")
            self.a = self.a_unit.base_value
            self.h_unit = self.get_unit("h")
            self.h = self.h_unit.base_value
            self.z = 1.0 / self.a - 1.0

            # Read cosmology
            self.cosmology = {}
            for name in infile["Cosmology"].attrs:
                self.cosmology[name] = infile["Cosmology"].attrs[name][0]

            # Read constants
            self.constants = {}
            for name in infile["PhysicalConstants"]["CGS"].attrs:
                self.constants[name] = infile["PhysicalConstants"]["CGS"].attrs[name][0]
            self.constants_internal = {}
            for name in infile["PhysicalConstants"]["InternalUnits"].attrs:
                self.constants_internal[name] = infile["PhysicalConstants"]["InternalUnits"].attrs[name][0]

            # Store units groups
            self.swift_units_group = {}
            for name in infile["Units"].attrs:
                self.swift_units_group[name] = infile["Units"].attrs[name][0]
            self.swift_internal_units_group = {}
            for name in infile["InternalCodeUnits"].attrs:
                self.swift_internal_units_group[name] = infile[
                    "InternalCodeUnits"
                ].attrs[name][0]

            # Store SWIFT header
            self.swift_header_group = {}
            for name in infile["Header"].attrs:
               self.swift_header_group[name] = infile["Header"].attrs[name]

            # Read the critical density and attach units
            # This is in internal units, which may not be the same as snapshot units.
            critical_density = float(
                self.cosmology["Critical density [internal units]"]
            )
            internal_density_unit = self.get_unit("code_mass") / (
                self.get_unit("code_length") ** 3
            )
            self.critical_density = unyt.unyt_quantity(
                critical_density, units=internal_density_unit
            )

            # Compute mean density at the redshift of the snapshot:
            # Here we compute the mean density in internal units at z=0 using
            # constants from the snapshot. The comoving mean density is
            # constant so we can then just scale by a**3 to get the physical
            # mean density.
            H0 = self.cosmology["H0 [internal units]"]
            G  = self.constants_internal["newton_G"]
            critical_density_z0_internal = 3*(H0**2) / (8*np.pi*G)
            mean_density_z0_internal = critical_density_z0_internal * self.cosmology["Omega_m"]
            mean_density_internal = mean_density_z0_internal / (self.a**3)
            self.mean_density = unyt.unyt_quantity(mean_density_internal, units=internal_density_unit)

            # Compute the BN98 critical density multiple
            Omega_k = self.cosmology["Omega_k"]
            Omega_Lambda = self.cosmology["Omega_lambda"]
            Omega_m = self.cosmology["Omega_m"]
            bnx = -(Omega_k / self.a**2 + Omega_Lambda) / (
                Omega_k / self.a**2 + Omega_m / self.a**3 + Omega_Lambda
            )
            self.virBN98 = 18.0 * np.pi**2 + 82.0 * bnx - 39.0 * bnx**2
            if self.virBN98 < 50.0 or self.virBN98 > 1000.0:
                raise RuntimeError("Invalid value for virBN98!")

            # Get the box size. Assume it's comoving with no h factors.
            comoving_length_unit = self.get_unit("snap_length") * self.a_unit
            self.boxsize = unyt.unyt_quantity(
                infile["Header"].attrs["BoxSize"][0], units=comoving_length_unit
            )

            # Get the observer position for the first lightcone
            try:
                observer_position_str = (
                    infile["Parameters"]
                    .attrs["Lightcone0:observer_position"]
                    .decode("utf-8")
                )
                observer_position = [
                    float(x) for x in observer_position_str[1:-1].split(",")
                ]
                self.observer_position = unyt.unyt_array(
                    observer_position, units=comoving_length_unit
                )
            except KeyError:
                print(
                    "Could not find lightcone observer position in snapshot file. Defaulting to centre of box."
                )
                self.observer_position = 0.5 * unyt.unyt_array([self.boxsize] * 3)

            # Get the number of files
            self.nr_files = infile["Header"].attrs["NumFilesPerSnapshot"][0]

            # Read cell meta data
            self.ptypes = []
            self.nr_cells = infile["Cells/Meta-data"].attrs["nr_cells"]
            self.dimension = infile["Cells/Meta-data"].attrs["dimension"]
            self.cell_size = unyt.unyt_array(
                infile["Cells/Meta-data"].attrs["size"], units=comoving_length_unit
            )
            for name in infile["Cells/Counts"]:
                self.ptypes.append(name)

            # Create arrays of cells
            self.cell = {}
            for ptype in self.ptypes:
                self.cell[ptype] = np.ndarray(self.nr_cells, dtype=swift_cell_t)

            # Read cell info
            for ptype in self.ptypes:
                cellgrid = self.cell[ptype]
                cellgrid["centre"] = infile["Cells/Centres"][...]
                cellgrid["count"] = infile["Cells"]["Counts"][ptype][...]
                cellgrid["offset"] = infile["Cells"]["OffsetsInFile"][ptype][...]
                cellgrid["file"] = infile["Cells"]["Files"][ptype][...]

        # Determine ordering of the cells in the snapshot
        for ptype in self.ptypes:
            cellgrid = self.cell[ptype]
            idx = np.lexsort((cellgrid["offset"], cellgrid["file"]))
            for cell_order, cell_index in enumerate(idx):
                cellgrid[cell_index]["order"] = cell_order

        # Reshape into a grid
        for ptype in self.ptypes:
            self.cell[ptype] = self.cell[ptype].reshape(self.dimension)

        # Scan files to find shape and dtype etc for all quantities in the snapshot.
        self.snap_metadata = identify_datasets(
            snap_filename, self.nr_files, self.ptypes, self.snap_unit_registry
        )
        if extra_filename is not None:
            self.extra_metadata = identify_datasets(
                extra_filename, self.nr_files, self.ptypes, self.snap_unit_registry
            )

        # Scan reference snapshot for missing particle types (e.g. stars or black holes at high z)
        self.ptypes_ref = {}
        if snap_filename_ref is not None:
            # Determine any particle types present in the reference snapshot but not in the current snapshot
            with h5py.File(snap_filename_ref % {"file_nr": 0}, "r") as infile:
                for name in infile["Cells/Counts"]:
                    if name not in self.ptypes:
                        self.ptypes_ref.append(name)
            # Scan reference snapshot for properties of additional particle types
            if len(self.ptypes_ref) > 0:
                self.snap_metadata_ref = identify_datasets(
                    snap_filename_ref,
                    self.nr_files,
                    self.ptypes_ref,
                    self.snap_unit_registry,
                )
                if extra_filename_ref is not None:
                    self.extra_metadata_ref = identify_datasets(
                        extra_filename_ref,
                        self.nr_files,
                        self.ptypes_ref,
                        self.snap_unit_registry,
                    )

    def prepare_read(self, ptype, mask):
        """
        Determine which ranges of particles we need to read from each file
        to read all of the cells indicated by the mask for the specified
        particle type.

        ptype - which particle type to read
        mask  - 3D boolean array with one element per cell, true if the
                cell is to be read and false otherwise

        Returns a dict where the keys are the unique file numbers to read
        and the values are lists of (offset_in_file, offset_in_memory, count) tuples.
        """

        # Make an array of the selected cells
        cells_to_read = self.cell[ptype][mask].flatten()

        # Discard any empty cells
        cells_to_read = cells_to_read[cells_to_read["count"] > 0]

        # Sort the selected cells by file, and then by offset within the file
        idx = np.lexsort((cells_to_read["offset"], cells_to_read["file"]))
        cells_to_read = cells_to_read[idx]

        # Find unique file numbers to read
        unique_file_nrs = np.unique(cells_to_read["file"])

        # Make a list of reads for each file:
        # reads[file_nr] is a list of (file_offset, memory_offset, count) tuples for file file_nr.
        mem_offset = 0
        reads = {file_nr: [] for file_nr in unique_file_nrs}
        for cell in cells_to_read:
            reads[cell["file"]].append((cell["offset"], mem_offset, cell["count"]))
            mem_offset += cell["count"]

        return reads

    def empty_mask(self):

        return np.zeros(self.dimension, dtype=bool)

    def mask_region(self, mask, pos_min, pos_max):
        imin = np.asarray(np.floor(pos_min / self.cell_size), dtype=int)
        imax = np.asarray(np.floor(pos_max / self.cell_size), dtype=int)
        for i in range(imin[0], imax[0] + 1):
            ii = i % self.dimension[0]
            for j in range(imin[1], imax[1] + 1):
                jj = j % self.dimension[1]
                for k in range(imin[2], imax[2] + 1):
                    kk = k % self.dimension[2]
                    mask[ii, jj, kk] = True

    def read_masked_cells_to_shared_memory(
        self, property_names, mask, comm, max_ranks_reading
    ):
        """
        Read in the specified properties for the cells with mask=True
        """

        comm_size = comm.Get_size()
        comm_rank = comm.Get_rank()

        # Make a list of all reads to execute for each particle type in this snapshot
        reads_for_type = {}
        for ptype in property_names:
            if ptype in self.ptypes:
                reads_for_type[ptype] = self.prepare_read(ptype, mask)

        # Find union of file numbers to read for all particle types
        all_file_nrs = []
        for ptype in reads_for_type:
            all_file_nrs += list(reads_for_type[ptype])
        all_file_nrs = np.unique(all_file_nrs)

        # Count particles to read in
        nr_parts = {ptype: 0 for ptype in reads_for_type}
        for file_nr in all_file_nrs:
            for ptype in reads_for_type:
                if file_nr in reads_for_type[ptype]:
                    for (file_offset, mem_offset, count) in reads_for_type[ptype][
                        file_nr
                    ]:
                        nr_parts[ptype] += count

        # Allocate MPI shared memory for the particle data for types which exist
        # in this snapshot. Note that these allocations could have zero size if
        # there are no particles of a type in the masked cells.
        data = {}
        for ptype in reads_for_type:
            data[ptype] = {}
            for name in property_names[ptype]:

                # Get metadata for array to allocate in memory
                if name in self.snap_metadata[ptype]:
                    units, dtype, shape = self.snap_metadata[ptype][name]
                elif (
                    self.extra_metadata is not None
                    and name in self.extra_metadata[ptype]
                ):
                    units, dtype, shape = self.extra_metadata[ptype][name]
                else:
                    raise Exception(
                        "Can't find required dataset %s in input file(s)!" % name
                    )

                # Determine size of local array section
                nr_local = nr_parts[ptype] // comm_size
                if comm_rank < (nr_parts[ptype] % comm_size):
                    nr_local += 1
                # Find global and local shape of the array
                global_shape = (nr_parts[ptype],) + shape
                local_shape = (nr_local,) + shape
                # Allocate storage
                data[ptype][name] = shared_array.SharedArray(
                    local_shape, dtype, comm, units
                )

        comm.barrier()

        # Divide communicator and assign different files to different communicators
        max_nr_io_comms = max_ranks_reading
        nr_io_comms = min(len(all_file_nrs), max_nr_io_comms)
        ranks_per_io_comm = math.ceil(comm_size / nr_io_comms)
        io_comm_nr = comm_rank // ranks_per_io_comm
        io_comm = comm.Split(color=io_comm_nr, key=comm_rank)
        io_comm_size = io_comm.Get_size()
        io_comm_rank = io_comm.Get_rank()
        nr_io_comms = comm.allreduce(io_comm_nr, op=MPI.MAX) + 1

        # Loop over files to read:
        # Snapshot files and, optionally, corresponding extra data files.
        for file_index, file_nr in enumerate(all_file_nrs):            
            for (filename, metadata) in ((self.snap_filename, self.snap_metadata),
                                         (self.extra_filename, self.extra_metadata)):

                # Skip files we're not reading
                if filename is None or (file_index % nr_io_comms) != io_comm_nr:
                    continue

                # Open the current file in MPI mode
                infile = h5py.File(filename % {"file_nr": file_nr}, "r", driver="mpio", comm=io_comm)

                # Loop over particle types to read
                for ptype in reads_for_type:

                    # Distribute read operations for this particle type between MPI ranks
                    all_reads = reads_for_type[ptype][file_nr]
                    reads_per_rank = np.zeros(io_comm_size, dtype=int)
                    reads_per_rank[:] = len(all_reads) // io_comm_size
                    reads_per_rank[:len(all_reads) % io_comm_size] += 1
                    assert sum(reads_per_rank) == len(all_reads)
                    first_on_rank = np.cumsum(reads_per_rank) - reads_per_rank
                    local_reads = all_reads[first_on_rank[io_comm_rank]:
                                            first_on_rank[io_comm_rank]+reads_per_rank[io_comm_rank]]

                    # Loop over properties to read
                    group = infile[ptype]
                    for name in property_names[ptype]:

                        if name not in metadata[ptype]:
                            continue

                        # Open the dataset
                        dataset = group[name]
                        file_shape = dataset.shape

                        # Get the file dataspace with no elements selected
                        file_space = dataset.id.get_space()
                        file_space.select_none()

                        # Select regions in the file which this rank should read
                        for (file_offset, mem_offset, count) in local_reads:
                            select_start = (file_offset,) + (0,)*len(file_shape[1:])
                            select_count = (count,) + file_shape[1:]
                            file_space.select_hyperslab(select_start, select_count, op=h5py.h5s.SELECT_OR)

                        # Create memory dataspace with no elements selected
                        mem_shape = (nr_parts[ptype],) + file_shape[1:]
                        mem_space = h5py.h5s.create_simple(mem_shape)
                        mem_space.select_none()

                        # Select regions in memory which this rank should read
                        ntot = 0
                        for (file_offset, mem_offset, count) in local_reads:
                            select_start = (mem_offset,) + (0,)*len(file_shape[1:])
                            select_count = (count,) + file_shape[1:]
                            mem_space.select_hyperslab(select_start, select_count, op=h5py.h5s.SELECT_OR)
                            ntot += count

                        # Do a collective read
                        dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
                        dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)
                        dataset.id.read(mem_space, file_space, data[ptype][name].full, dxpl=dxpl)

                        # Tidy up and move on to the next dataset
                        dxpl.close()
                        mem_space.close()
                        file_space.close()

                # Next file
                infile.close()

        # Create empty arrays for particle types which exist in the reference snapshot but not this one
        for ptype in property_names:
            if ptype in self.ptypes_ref:
                data[ptype] = {}
                for name, (units, dtype, shape) in self.snap_metadata_ref[
                    ptype
                ].items():
                    local_shape = (0,) + shape
                    data[ptype][name] = shared_array.SharedArray(
                        local_shape, dtype, comm, units
                    )
                for name, (units, dtype, shape) in self.extra_metadata_ref[
                    ptype
                ].items():
                    local_shape = (0,) + shape
                    data[ptype][name] = shared_array.SharedArray(
                        local_shape, dtype, comm, units
                    )

        # Ensure all arrays have been fully written
        comm.barrier()
        for ptype in reads_for_type:
            for name in property_names[ptype]:
                data[ptype][name].sync()
        comm.barrier()

        io_comm.Free()

        return data

    def write_metadata(self, group):
        """
        Write simulation information etc to the specified HDF5 group
        """

        # Write cosmology
        cosmo = group.create_group("Cosmology")
        for name, value in self.cosmology.items():
            cosmo.attrs[name] = [
                value,
            ]

        # Write physical constants
        const = group.create_group("PhysicalConstants")
        const = const.create_group("CGS")
        for name, value in self.constants.items():
            const.attrs[name] = [
                value,
            ]

        # Write units
        units = group.create_group("Units")
        for name, value in self.swift_units_group.items():
            units.attrs[name] = [
                value,
            ]
        units = group.create_group("InternalCodeUnits")
        for name, value in self.swift_internal_units_group.items():
            units.attrs[name] = [
                value,
            ]

        # Write header
        header = group.create_group("Header")
        for name, value in self.swift_header_group.items():
            header.attrs[name] = value

    def complete_radius_from_mask(self, mask):
        """
        Given a mask of selected cells, for each selected cell compute
        a radius within which we are guaranteed to have read all particles
        around any halo that could exist in the cell.

        Here we assume that cells can contribute particles up to half a cell
        outside their own volume, so the furthest a particle can be from the
        centre of its parent cell is equal to the cell diagonal. In the worst
        case we can have a halo at the corner of its parent cell nearest to
        a cell which has not been read. Then the radius within which we have
        all particles is limited to the distance between the cell centres
        minus 1.5 times the cell diagonal.

        This is used to handle the case where we didn't ask for a large enough
        radius around a halo: it may be that we still have enough particles in
        memory due to reading cells needed for adjacent halos.
        """

        # All types use the same grid, so just use cell arrays for the first type
        ptype = list(self.cell.keys())[0]
        cell_centre = self.cell[ptype]["centre"]
        cell_diagonal = np.sqrt(np.sum(self.cell_size.value ** 2))

        # Output array
        cell_complete_radius = np.zeros(self.dimension)

        # Make tree with the centers of the cells we did not read
        centre_not_read = cell_centre[mask == False, :]
        tree = scipy.spatial.cKDTree(centre_not_read, boxsize=self.boxsize.value)

        # For each cell, find the nearest cell we didn't read
        distance, index = tree.query(cell_centre, 1)
        cell_complete_radius[:, :, :] = distance.reshape(mask.shape)

        # Get a limit on the radius within which halos in the cell have all particles
        cell_complete_radius -= 1.5 * cell_diagonal
        cell_complete_radius[cell_complete_radius < 0.0] = 0.0

        # Return the result in suitable units
        comoving_length_unit = self.get_unit("snap_length") * self.a_unit
        return unyt.unyt_array(cell_complete_radius, units=comoving_length_unit)
