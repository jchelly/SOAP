#!/bin/env python

import time

import numpy as np
import unyt

from dataset_names import mass_dataset, ptypes_for_so_masses
from halo_properties import ReadRadiusTooSmallError
import shared_array
import result_set
import halo_properties

# Factor by which to increase search radius when looking for density threshold
SEARCH_RADIUS_FACTOR=1.2

# Factor by which to increase the region read in around a halo if too small
READ_RADIUS_FACTOR=1.5

# Radius in Mpc at which we report halos which have a large search radius
REPORT_RADIUS=20.0


def process_single_halo(mesh, unit_registry, data, halo_prop_list,
                        critical_density, mean_density, boxsize, input_halo,
                        target_density):
    """
    This computes properties for one halo and runs on a single
    MPI rank. Result is a dict of properties of the form
    
    halo_result[property_name] = (unyt_array, description)

    where the property_name will be used as the HDF5 dataset name
    in the output and the units of the unyt_array determine the unit
    attributes.

    Two radii are passed in:

    search_radius is an initial guess at the radius we need to fall
    below the specified overdensity threshold. read_radius is the
    radius within which we have all of the particles. If we find that
    the density within read_radius is above the threshold, then we
    didn't read in a large enough region.

    Returns None if we need to try again with a larger region.
    """

    swift_mpc   = unyt.Unit("swift_mpc", registry=unit_registry)    
    snap_length = unyt.Unit("snap_length", registry=unit_registry)
    snap_mass   = unyt.Unit("snap_mass", registry=unit_registry)
    snap_density = snap_mass / (snap_length**3)

    # Record which calculations are still to do for this halo
    halo_prop_done = np.zeros(len(halo_prop_list), dtype=bool)

    # Dict to store the results
    halo_result = {}

    # Loop until we fall below the required density
    current_radius = input_halo["search_radius"]
    while True:
        
        # Sanity checks on the radius
        assert current_radius <= input_halo["read_radius"]
        if current_radius > REPORT_RADIUS*swift_mpc:
            print("Halo ID={input_halo['ID']} has large search radius {current_radius}")

        # Find the mass within the search radius
        mass_total = unyt.unyt_quantity(0.0, units=snap_mass)
        idx = {}
        for ptype in data:
            pos = data[ptype]["Coordinates"]
            idx[ptype] = mesh[ptype].query_radius_periodic(input_halo["cofp"], current_radius, pos, boxsize)
            if ptype in ptypes_for_so_masses:
                mass = data[ptype][mass_dataset(ptype)]
                mass_total += np.sum(mass.full[idx[ptype]], dtype=float)

        # Find mean density in the search radius
        density = mass_total / (4./3.*np.pi*current_radius**3)

        # If we've reached the target density, we can try to compute halo properties
        max_physical_radius_mpc = 0.0 # Will store the largest physical radius requested by any calculation
        if target_density is None or density <= target_density:

            # Extract particles in this halo
            particle_data = {}
            for ptype in data:
                particle_data[ptype] = {}
                for name in data[ptype]:
                    particle_data[ptype][name] = data[ptype][name].full[idx[ptype],...]

            # Wrap coordinates to copy closest to the halo centre
            for ptype in particle_data:
                pos = particle_data[ptype]["Coordinates"]
                # Shift halo to box centre, wrap all particles into box, shift halo back
                offset = input_halo["cofp"] - 0.5*boxsize
                pos[:,:] = ((pos - offset) % boxsize) + offset

            # Sort particles by radius
            for ptype in particle_data:
                pos = particle_data[ptype]["Coordinates"]
                r = np.linalg.norm(pos - input_halo["cofp"][None,:], axis=1)
                order = np.argsort(r)
                for name in particle_data[ptype]:
                    particle_data[ptype][name] = particle_data[ptype][name][order,...]
                assert "Radius" not in particle_data[ptype]
                particle_data[ptype]["Radius"] = r

            # Try to compute properties of this halo which haven't been done yet
            for prop_nr, halo_prop in enumerate(halo_prop_list):
                if halo_prop_done[prop_nr]:
                    # Already have the result for this one
                    continue
                try:
                    halo_prop.calculate(input_halo, current_radius, particle_data, halo_result)
                except ReadRadiusTooSmallError:
                    # Search radius was too small, so will need to try again with a larger radius.
                    max_physical_radius_mpc = max(max_physical_radius_mpc, halo_prop.physical_radius_mpc)
                    break
                else:
                    # The property calculation worked!
                    halo_prop_done[prop_nr] = True

            # If we computed all of the properties, we're done with this halo
            if np.all(halo_prop_done):
                break

        # Either the density is still too high or the property calculation failed.
        search_radius = input_halo["search_radius"]
        # To improve reproducibility, avoid setting the search radius exactly equal to the
        # radius needed for the calculations so that particles right on the boundary don't
        # get included/excluded depending on rounding error. Here we add a 1% safety factor.        
        required_radius = (1.01*max_physical_radius_mpc*swift_mpc).to(search_radius.units)
        if required_radius > input_halo["read_radius"]:
            # A calculation has set its physical_radius_mpc larger than the region
            # which we read in, so we can't process the halo on this iteration regardless
            # of the current radius.
            input_halo["search_radius"] = max(search_radius, required_radius)
            return None
        elif current_radius >= input_halo["read_radius"]:
            # The current radius has exceeded the region read in. Will need to redo this
            # halo using current_radius as the starting point for the next iteration.
            input_halo["search_radius"] = max(search_radius, current_radius)
            return None
        else:
            # We still have a large enough region in memory that we can try a larger radius
            current_radius = min(current_radius*SEARCH_RADIUS_FACTOR, input_halo["read_radius"])
            current_radius = max(current_radius, required_radius)

    # Add the halo index to the result set
    halo_result["VR/index"]         = (input_halo["index"],         "Index of this halo in the input catalogue")
    halo_result["VR/ID"]            = (input_halo["ID"],            "VELOCIraptor halo ID")
    halo_result["VR/Structuretype"] = (input_halo["Structuretype"], "VELOCIraptor Structuretype parameter")

    # Store search radius and density within that radius
    halo_result["SearchRadius/search_radius"]            = (current_radius,                  "Search radius for property calculation")
    halo_result["SearchRadius/density_in_search_radius"] = (density.to(snap_density),        "Density within the search radius")
    halo_result["SearchRadius/target_density"]           = (target_density.to(snap_density), "Target density for property calculation")

    return halo_result


def process_halos(comm, unit_registry, data, mesh, halo_prop_list,
                  critical_density, mean_density, boxsize, halo_arrays):
    """
    This uses all of the MPI ranks on one compute node to compute halo properties
    for a single "chunk" of the simulation.

    Each rank returns a dict with the properties of the halos it processed.
    The dict keys are the property names. The values are tuples containing
    (unyt_array, description).
    
    Halos where done=1 are not processed and don't generate an entry in the
    output arrays.

    If a halo can't be processed because we didn't read enough particles,
    its read_radius is doubled but no entry is generated in the output.
    """
    # Compute density threshold at this redshift in comoving units:
    # This determines the size of the sphere we use for all other SO quantities.
    # We need to find the minimum density required for any of the halo property
    # calculations
    target_density = None
    for halo_prop in halo_prop_list:
        # Ensure target density is no greater than mean density multiple
        if halo_prop.mean_density_multiple is not None:
            density = halo_prop.mean_density_multiple*mean_density
            if target_density is None or density < target_density:
                target_density = density
        # Ensure target density is no greater than critical density multiple
        if halo_prop.critical_density_multiple is not None:
            density = halo_prop.critical_density_multiple*critical_density
            if target_density is None or density < target_density:
                target_density = density

    # Allocate shared storage for a single integer and initialize to zero
    if comm.Get_rank() == 0:
        local_shape = (1,)
    else:
        local_shape = (0,)
    next_task = shared_array.SharedArray(local_shape, np.int64, comm)
    if comm.Get_rank() == 0:
        next_task.full[0] = 0
    next_task.sync()

    # Make a ResultSet to store the output
    all_results = result_set.ResultSet()

    # Start the clock
    comm.barrier()
    t0_all = time.time()

    # Count halos to do
    nr_halos_left = comm.allreduce(np.sum(halo_arrays["done"].local==0))

    # Loop until all halos are done
    result_arrays = {}
    nr_halos = len(halo_arrays["index"].full)
    nr_done_this_rank = 0
    nr_halos_this_rank_guess = int(nr_halos_left / comm.Get_size() * 1.5)
    task_time = 0.0
    while True:

        # Get a task by atomic incrementing the counter. Don't know how to do
        # an atomic fetch and add in python, so will use MPI RMA calls!
        task_to_do = np.ndarray(1, dtype=np.int64)
        one = np.ones(1, dtype=np.int64)
        next_task.win.Lock(0)
        next_task.win.Fetch_and_op(one, task_to_do, 0)
        next_task.win.Unlock(0)
        task_to_do = int(task_to_do)

        # Execute the task, if there's one left
        if task_to_do < nr_halos:
            t0_task = time.time()

            # Skip halos we already did
            if halo_arrays["done"].full[task_to_do] == 0:

                # Extract this halo's VR information (centre, radius, index etc)
                input_halo = {}
                for name in halo_arrays:
                    input_halo[name] = halo_arrays[name].full[task_to_do,...].copy()

                # Fetch the results for this particular halo
                results = process_single_halo(mesh, unit_registry, data, halo_prop_list,
                                              critical_density, mean_density,
                                              boxsize, input_halo, target_density)
                if results is not None:
                    # Store results and flag this halo as done
                    all_results.append(results)
                    nr_done_this_rank += 1
                    halo_arrays["done"].full[task_to_do] = 1
                else:
                    # We didn't read in a large enough region. Update the shared radius
                    # arrays so that we read a larger region next time and start the
                    # search from whatever radius we had reached this time.
                    new_read_radius = max(input_halo["read_radius"] * READ_RADIUS_FACTOR,
                                          input_halo["search_radius"])
                    # Set the radius around the halo to read in
                    halo_arrays["read_radius"].full[task_to_do] = new_read_radius
                    # Set the initial guess at the radius we need
                    halo_arrays["search_radius"].full[task_to_do] = input_halo["search_radius"]

            t1_task = time.time()
            task_time += (t1_task-t0_task)
        else:
            # We ran out of halos to do
            break

    # Resize output arrays, since they may have been allocated larger than needed
    all_results.trim()

    # Free the shared task counter
    next_task.free()

    # Count halos left to do
    comm.barrier()
    nr_halos_left = comm.allreduce(np.sum(halo_arrays["done"].local==0))

    # Stop the clock
    comm.barrier()
    t1_all = time.time()
    
    return all_results, t1_all-t0_all, task_time, nr_halos_left, comm.allreduce(nr_done_this_rank)
