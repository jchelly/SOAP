#!/bin/env python

import numpy as np
import unyt

from dataset_names import mass_dataset


class HaloProperty:

    def __init__(self, cellgrid):

        # Store parameters needed for halo property calculations
        self.unit_registry    = cellgrid.snap_unit_registry # unyt registry with snapshot units
        self.critical_density = cellgrid.critical_density   # critical density as unyt_quantity
        self.mean_density     = cellgrid.mean_density       # mean density as unyt_quantity
        self.a                = cellgrid.a                  # expansion factor of this snapshot
        self.a_unit           = cellgrid.a_unit             # Dimensionless unit used to define comoving quantities
        self.z                = cellgrid.z                  # redshift of this snapshot
        self.boxsize          = cellgrid.boxsize            # boxsize as unyt_quantity


class SOMasses(HaloProperty):
    
    # Name of this calculation, used to select on the command line
    name="so_masses"

    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0" : ["Coordinates", "Masses"],
        "PartType1" : ["Coordinates", "Masses"],
        "PartType4" : ["Coordinates", "Masses"],
        "PartType5" : ["Coordinates", "DynamicalMasses"]
    }

    # This specifies how large a sphere is read in:
    # Will ensure we have a sphere with a mean density less than
    # or equal to the minimum of these densities.
    mean_density_multiple     = 200.0
    critical_density_multiple = 200.0

    # Minimum physical radius to read in (pMpc)
    physical_radius_mpc = 0.0

    def calculate(self, input_halo, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Look up VR centre of potential for this halo
        centre = input_halo["cofp"]

        # Make an array of particle masses and radii
        mass = []
        radius = []
        for ptype in data:
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None,:]
            r = np.sqrt(np.sum(pos**2, axis=1))
            radius.append(r)
        mass = unyt.array.uconcatenate(mass)
        radius = unyt.array.uconcatenate(radius)
        nr_parts = mass.shape[0]

        # Sort by radius
        order = np.argsort(radius)
        mass = mass[order]
        radius = radius[order]
        cumulative_mass = np.cumsum(mass, dtype=np.float64)

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        nskip = 0
        while nskip < len(radius) and radius[nskip] == 0:
            nskip += 1
        radius  = radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(radius)
        density = cumulative_mass / (4./3.*np.pi*radius**3)

        # Check if we ever reach the density threshold
        if nr_parts > 0 and np.any(density > 200*self.critical_density):
            # Find smallest radius where the density is below the threshold
            i = np.argmax(density < 200*self.critical_density)
            m200crit = cumulative_mass[i]
            r200crit = radius[i]
        else:
            # Below threshold at all radii. Need to return zero with correct units attached.
            m200crit = unyt.unyt_array(0, dtype=cumulative_mass.dtype, units=cumulative_mass.units)
            r200crit = unyt.unyt_array(0, dtype=radius.dtype, units=radius.units)

        # Update this halo's properties
        halo_result.update({
            "r_200_crit" : (r200crit, "Radius within which the density is 200 times the mean"),
            "m_200_crit" : (m200crit, "Mass within a sphere with density 200 times the mean"),
        })


class SubhaloBoundMasses(HaloProperty):

    # Name of this calculation, used to select on the command line
    name="subhalo_bound_masses"
    
    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0" : ["Coordinates", "Velocities", "Masses", "GroupNr_bound"],
        "PartType1" : ["Coordinates", "Velocities", "Masses", "GroupNr_bound"],
        "PartType4" : ["Coordinates", "Velocities", "Masses", "InitialMasses", "GroupNr_bound"],
        "PartType5" : ["Coordinates", "Velocities", "DynamicalMasses", "SubgridMasses", "GroupNr_bound"]
    }

    # This specifies how large a sphere is read in:
    # Will ensure we have a sphere with a mean density less than
    # or equal to the minimum of these densities.
    mean_density_multiple     = None
    critical_density_multiple = None

    # Minimum physical radius to read in (pMpc)
    physical_radius_mpc = 0.0

    def calculate(self, input_halo, data, halo_result):
        """
        Compute centre of mass of bound particles

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """
        
        # Storage for properties of each particle type
        cofm_pos   = {}
        cofm_vel   = {}
        total_mass = {}
        nr_part    = {}

        # Look up array index of this halo in VR catalogue
        index = input_halo["index"]

        # Find the simulation mass unit
        mass_unit = unyt.Unit("snap_mass",   registry=self.unit_registry)

        # Loop over particle types
        nr_part_all = unyt.unyt_quantity(0.0, units=unyt.dimensionless, registry=self.unit_registry)
        total_mass_all = unyt.unyt_quantity(0.0, units=mass_unit)
        total_initial_mass = unyt.unyt_quantity(0.0, units=mass_unit)
        total_subgrid_mass = unyt.unyt_quantity(0.0, units=mass_unit)
        for ptype in data:

            # Find position and mass of particles in the group
            grnr = data[ptype]["GroupNr_bound"]
            in_halo = (grnr==index)
            pos  = data[ptype]["Coordinates"][in_halo,:]
            vel  = data[ptype]["Velocities"][in_halo,:]
            mass = data[ptype][mass_dataset(ptype)][in_halo]

            # Store total mass of particles of this type
            total_mass[ptype] = np.sum(mass, dtype=float)
            total_mass_all += total_mass[ptype]

            # Store centre of mass and velocity for particles of this type
            if total_mass[ptype] > 0.0:
                cofm_pos[ptype] = np.sum(pos*mass[:,None], axis=0, dtype=float) / total_mass[ptype]
                cofm_vel[ptype] = np.sum(vel*mass[:,None], axis=0, dtype=float) / total_mass[ptype]
            else:
                cofm_pos[ptype] = unyt.unyt_quantity(0.0, units=pos.units, dtype=float)
                cofm_vel[ptype] = unyt.unyt_quantity(0.0, units=vel.units, dtype=float)
        
            # Accumulate total number of particles of this type
            nr_part[ptype] = unyt.unyt_quantity(pos.shape[0], units=unyt.dimensionless, dtype=int)
            nr_part_all += nr_part[ptype]

            # Total stellar initial mass
            if ptype == "PartType4":
                total_initial_mass = np.sum(data[ptype]["InitialMasses"][in_halo], dtype=float)

            # Total BH subgrid mass
            if ptype == "PartType5":
                total_subgrid_mass = np.sum(data[ptype]["SubgridMasses"][in_halo], dtype=float)

        # Box wrap the positions
        for ptype in cofm_pos:
            cofm_pos[ptype] = cofm_pos[ptype] % self.boxsize

        # Add these properties to the output
        for ptype in nr_part:
            halo_result["NumPart_"+ptype]              = (nr_part[ptype],     "Number of particles of type "+ptype)
            halo_result["Mass_"+ptype]                 = (total_mass[ptype],  "Total mass of particles of type "+ptype)
            halo_result["CentreOfMass_"+ptype]         = (cofm_pos[ptype],    "Centre of mass of particles of type "+ptype)
            halo_result["CentreOfMassVelocity_"+ptype] = (cofm_vel[ptype],    "Centre of mass velocity of particles of type "+ptype)
        halo_result["StellarInitialMass"]              = (total_initial_mass, "Total initial mass of star particles")
        halo_result["BHSubgridMass"]                   = (total_subgrid_mass, "Total subgrid mass of black hole particles")
        halo_result["Mass_All"]                        = (total_mass_all,     "Total subgrid mass of black hole particles")
        halo_result["NumPart_All"]                     = (nr_part_all,        "Total number of particles")
