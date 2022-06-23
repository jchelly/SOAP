#!/bin/env python

import numpy as np
import unyt
from scipy.optimize import brentq

from halo_properties import HaloProperty, ReadRadiusTooSmallError

from dataset_names import mass_dataset

from mpi4py import MPI


def cumulative_mass_and_density(input_halo, nu_density, data):

    centre = input_halo["cofp"]

    # Make an array of particle masses, radii and positions
    mass = []
    radius = []
    position = []
    velocity = []
    types = []
    groupnr = []
    for ptype in data:
        if ptype == "PartType6":
            continue
        mass.append(data[ptype][mass_dataset(ptype)])
        pos = data[ptype]["Coordinates"] - centre[None, :]
        position.append(pos)
        r = np.sqrt(np.sum(pos**2, axis=1))
        radius.append(r)
        velocity.append(data[ptype]["Velocities"])
        typearr = np.zeros(r.shape, dtype="U9")
        typearr[:] = ptype
        types.append(typearr)
        groupnr.append(data[ptype]["GroupNr_bound"])
    mass = unyt.array.uconcatenate(mass)
    radius = unyt.array.uconcatenate(radius)
    position = unyt.array.uconcatenate(position)
    velocity = unyt.array.uconcatenate(velocity)
    types = np.concatenate(types)
    groupnr = unyt.array.uconcatenate(groupnr)

    # figure out which particles in the list are bound to a halo that is not the
    # central halo
    is_bound_to_satellite = (groupnr >= 0) & (groupnr != input_halo["index"])

    # add neutrinos
    if "PartType6" in data:
        numass = data["PartType6"]["Masses"] * data["PartType6"]["Weights"]
        pos = data["PartType6"]["Coordinates"] - centre[None, :]
        nur = np.sqrt(np.sum(pos**2, axis=1))
        all_mass = unyt.array.uconcatenate([mass, numass])
        all_r = unyt.array.uconcatenate([radius, nur])
    else:
        all_mass = mass
        all_r = radius

    # Sort by radius
    order = np.argsort(all_r)
    ordered_radius = all_r[order]
    cumulative_mass = np.cumsum(all_mass[order], dtype=np.float64).astype(
        mass.dtype
    )
    # add mean neutrino mass
    cumulative_mass += nu_density * 4.0 / 3.0 * np.pi * ordered_radius**3

    # Compute density within radius of each particle.
    # Will need to skip any at zero radius.
    # Note that because of the definition of the centre of potential, the first
    # particle *should* be at r=0. We need to manually exclude it, in case round
    # off error places it at a very small non-zero radius.
    nskip = max(1, np.argmax(ordered_radius > 0.0 * ordered_radius.units))
    ordered_radius = ordered_radius[nskip:]
    cumulative_mass = cumulative_mass[nskip:]
    nr_parts = len(ordered_radius)
    density = cumulative_mass / (4.0 / 3.0 * np.pi * ordered_radius**3)

    return nr_parts, ordered_radius, density, cumulative_mass, radius, types, is_bound_to_satellite, position, velocity, mass

    
def cumulative_mass_intersection(r, rho_dim, slope_dim):
    """
    Function used to find the intersection of the cumulative mass curve at fixed
    mean density, and the actual cumulative mass as obtained from linear
    interpolation on a cumulative mass profile.

    The equation we want to solve is:
      4*pi/3*rho * r^3 - (M2-M1)/(r2-r1) * r + (M2-M1)/(r2-r1)*r1 - M1 = 0
    Since all quantities have units and scipy cannot handle those, we actually
    solve
      4*pi/3*rho_d * u^3 - S_d * u + S_d - 1 = 0,
    with
      rho_d = rho * r1**3 / M1
      S_d = (M2-M1)/(r2-r1) * (r1/M1)
    The result then needs to be multiplied with r1 to get the intersection radius
    """
    return 4.0 * np.pi / 3.0 * rho_dim * r**3 - slope_dim * r + slope_dim - 1.0


def find_SO_radius_and_mass(
    ordered_radius, density, cumulative_mass, reference_density
):
    """
    Find the radius and mass of an SO from the ordered density and cumulative
    mass profiles.

    The profiles are constructed by sorting the particles within the spherical
    region and then summing their masses in that order (assigning the full
    mass of the particle to the particle's radius). The density for every radius
    is then computed by dividing the cumulative mass profile by the volume of a
    sphere with that radius.

    The SO radius is defined as the radius at which the density profile dips
    below the given reference density. Unfortunately, real density profiles
    are noisy and can sometimes fluctuate around the threshold. We therefore
    define the SO radius as the first radius for which this happens, at machine
    precision.
    If no particles are below the threshold, then we raise an error and force an
    increase of the search radius.
    If all particles are below the threshold, we assume that the cumulative
    mass profile of the halo is linear out to the radius of the first particle,
    and then use the corresponding density profile to find the intersection.
    In all other cases, we find the actual SO radius by assuming a linear
    cumulative mass profile in the bin where the density dips below the
    threshold, and intersecting the corresponding density profile with the
    threshold. This approach requires a root finding algorithm and does not
    yield exactly the same result as linear interpolation in r-log(rho) space
    for the same bin (which is for example used by VELOCIraptor). It however
    guarantees that the SO mass we find is contained within the intersecting
    bin, which would otherwise not necessarily be true (especially if the
    intersecting bin is relatively wide). We could also interpolate both the
    radius and mass, but then the mean density of the SO would not necessarily
    match the target density, which is also weird.
    """

    # Compute a mask that marks particles above the threshold. We do this
    # exactly once.
    above_mask = density > reference_density
    if np.any(above_mask):
        # Get the complementary mask of particles below the threshold.
        # By using the complementary, we avoid any ambiguity about '>' vs '<='
        below_mask = ~above_mask
        # Find smallest radius where the density is below the threshold
        i = np.argmax(below_mask)
        if i == 0:
            if below_mask[i]:
                # we know that there are points above the threshold
                # unfortunately, the centre is not
                # find the next one that is:
                offset = np.argmax(above_mask)
                # now get the next point below the threshold relative w.r.t. this point
                i = np.argmax(below_mask[offset:])
                # +offset because i is now relative w.r.t. offset
                i += offset
            else:
                # 'i==0' can also mean no particles are below the threshold
                # in this case, we need to increase the search radius
                if ordered_radius[-1] > 20.0 * unyt.Mpc:
                    raise RuntimeError(
                        "Cannot find SO radius, but search radius is already larger than 20 Mpc!"
                    )
                raise ReadRadiusTooSmallError(
                    "SO radius multiple estimate was too small!"
                )
    else:
        # all non-zero radius particles are below the threshold
        # we linearly interpolate the mass from 0 to the particle radius
        # and determine the radius at which this interpolation matches the
        # target density
        # This is simply the solution of
        #    4*pi/3*r^3*rho = M[0]/r[0]*r
        # Note that if masses are allowed to be negative, the first cumulative
        # mass value could be negative. We make sure to avoid this problem
        ipos = 0
        while ipos < len(cumulative_mass) and cumulative_mass[ipos] < 0.0:
            ipos += 1
        if ipos == len(cumulative_mass):
            raise RuntimeError("Should never happen!")
        SO_r = np.sqrt(
            0.75
            * cumulative_mass[ipos]
            / (np.pi * ordered_radius[ipos] * reference_density)
        )
        SO_mass = cumulative_mass[ipos] * SO_r / ordered_radius[ipos]
        return SO_r, SO_mass, 4.0 * np.pi / 3.0 * SO_r**3

    # We now have the intersecting interval. Get the limits.
    r1 = ordered_radius[i - 1]
    r2 = ordered_radius[i]
    M1 = cumulative_mass[i - 1]
    M2 = cumulative_mass[i]
    # deal with the pathological case where r1==r2
    # we also need an interval where the density intersects
    while r1 == r2 or (above_mask[i - 1] == above_mask[i]):
        i += 1
        # if we run out of 'i', we need to increase the search radius
        if i >= len(density):
            if ordered_radius[-1] > 20.0 * unyt.Mpc:
                raise RuntimeError(
                    "Cannot find SO radius, but search radius is already larger than 20 Mpc!"
                )
            raise ReadRadiusTooSmallError("SO radius multiple estimate was too small!")
        # take the next interval
        r1 = r2
        r2 = ordered_radius[i]
        M1 = M2
        M2 = cumulative_mass[i]

    # compute the dimensionless quantities that enter the intersection equation
    # remember, we are simply solving
    #  4*pi/3*r^3*rho = M1 + (M2-M1)/(r2-r1)*(r-r1)
    rho_dim = reference_density * r1**3 / M1
    slope_dim = (M2 - M1) / (r2 - r1) * (r1 / M1)
    SO_r = r1 * brentq(
        cumulative_mass_intersection, 1.0, r2 / r1, args=(rho_dim, slope_dim)
    )

    SO_volume = 4.0 / 3.0 * np.pi * SO_r**3
    # compute the SO mass by requiring that the mean density in the SO is the
    # target density
    SO_mass = SO_volume * reference_density

    return SO_r, SO_mass, SO_volume


class SOProperties(HaloProperty):

    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0": [
            "ComptonYParameters",
            "Coordinates",
            "Densities",
            "GroupNr_bound",
            "Masses",
            "MetalMassFractions",
            "Pressures",
            "Temperatures",
            "Velocities",
            "XrayLuminosities",
            "XrayPhotonLuminosities",
        ],
        "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
            "GroupNr_bound",
            "InitialMasses",
            "Luminosities",
            "Masses",
            "MetalMassFractions",
            "Velocities",
        ],
        "PartType5": [
            "AccretionRates",
            "Coordinates",
            "DynamicalMasses",
            "GroupNr_bound",
            "LastAGNFeedbackScaleFactors",
            "ParticleIDs",
            "SubgridMasses",
            "Velocities",
        ],
        "PartType6": ["Coordinates", "Masses", "Weights"],
    }

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    #                 Should contain a "{label}" entry that will be adjusted to describe the sphere for this
    #                 particular type of SO.
    SO_properties = [
        # global properties
        ("r", 1, np.float32, "Mpc", "Radius of a sphere {label}"),
        ("mass", 1, np.float32, "Msun", "Mass within a sphere {label}"),
        (
            "N",
            1,
            np.uint32,
            "dimensionless",
            "Number of particles within a sphere {label}",
        ),
        ("com", 3, np.float32, "Mpc", "Centre of mass within a sphere {label}"),
        (
            "vcom",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity within a sphere {label}",
        ),
        (
            "Mfrac_satellites",
            1,
            np.float32,
            "dimensionless",
            "Fraction of mass that is bound to a satellite within a sphere {label}",
        ),
        # gas properties
        ("mass_gas", 1, np.float32, "Msun", "Total gas mass within a sphere {label}"),
        (
            "Jgas",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of gas within a sphere {label}",
        ),
        (
            "com_gas",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of gas within a sphere {label}",
        ),
        (
            "vcom_gas",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of gas within a sphere {label}",
        ),
        (
            "veldisp_gas",
            6,
            np.float32,
            "km**2/s**2",
            "Velocity dispersion of gas within a sphere {label}. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        (
            "Mgasmetal",
            1,
            np.float32,
            "Msun",
            "Total metal mass of gas within a sphere {label}",
        ),
        (
            "Mhotgas",
            1,
            np.float32,
            "Msun",
            "Total mass of gas with T > 1e5 K within a sphere {label}",
        ),
        (
            "Tgas",
            1,
            np.float32,
            "K",
            "Mass-weighted average temperature of gas with T > 1e5 K within a sphere {label}",
        ),
        (
            "Xraylum",
            3,
            np.float64,
            "erg/s",
            "Total Xray luminosity within a sphere {label}",
        ),
        (
            "Xrayphlum",
            3,
            np.float64,
            "1/s",
            "Total Xray photon luminosity within a sphere {label}",
        ),
        ("compY", 1, np.float64, "cm**2", "Total Compton y within a sphere {label}"),
        (
            "Ekin_gas",
            1,
            np.float64,
            "erg",
            "Total kinetic energy of the gas within a sphere {label}. Measured relative to the centre of mass velocity of all particles.",
        ),
        (
            "Etherm_gas",
            1,
            np.float64,
            "erg",
            "Total thermal energy of the gas within a sphere {label}",
        ),
        # DM properties
        ("mass_dm", 1, np.float32, "Msun", "Total DM mass within a sphere {label}"),
        (
            "JDM",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of DM within a sphere {label}",
        ),
        (
            "veldisp_dm",
            6,
            np.float32,
            "km**2/s**2",
            "Velocity dispersion of DM within a sphere {label}. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        # stellar properties
        (
            "mass_star",
            1,
            np.float32,
            "Msun",
            "Total stellar mass within a sphere {label}",
        ),
        (
            "com_star",
            3,
            np.float32,
            "Mpc",
            "Centre of mass of stars within a sphere {label}",
        ),
        (
            "vcom_star",
            3,
            np.float32,
            "km/s",
            "Centre of mass velocity of stars within a sphere {label}",
        ),
        (
            "veldisp_star",
            6,
            np.float32,
            "km**2/s**2",
            "Velocity dispersion of stars within a sphere {label}. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        (
            "Jstar",
            3,
            np.float32,
            "Msun*kpc*km/s",
            "Total angular momentum of stars within a sphere {label}",
        ),
        (
            "Mstarinit",
            1,
            np.float32,
            "Msun",
            "Total initial stellar mass with a sphere {label}",
        ),
        (
            "Mstarmetal",
            1,
            np.float32,
            "Msun",
            "Total metal mass of stars within a sphere {label}",
        ),
        (
            "Lstar",
            9,
            np.float32,
            "dimensionless",
            "Total stellar luminosity within a sphere {label}",
        ),
        # BH properties
        (
            "MBHdyn",
            1,
            np.float32,
            "Msun",
            "Total dynamical BH mass within a sphere {label}",
        ),
        (
            "MBHsub",
            1,
            np.float32,
            "Msun",
            "Total sub-grid BH mass within a sphere {label}",
        ),
        (
            "BHlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Last AGN feedback event within a sphere {label}",
        ),
        ("BHmaxM", 1, np.float32, "Msun", "Maximum BH mass within a sphere {label}"),
        (
            "BHmaxID",
            1,
            np.uint64,
            "dimensionless",
            "ID of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxpos",
            3,
            np.float32,
            "Mpc",
            "Position of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxvel",
            3,
            np.float32,
            "km/s",
            "Velocity of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxAR",
            1,
            np.float32,
            "Msun/yr",
            "Accretion rate of most massive BH within a sphere {label}",
        ),
        (
            "BHmaxlasteventa",
            1,
            np.float32,
            "dimensionless",
            "Last AGN feedback event of the most massive BH within a sphere {label}",
        ),
        # Neutrino properties
        (
            "MnuNS",
            1,
            np.float32,
            "Msun",
            "Noise suppressed total neutrino mass within a sphere {label}",
        ),
        (
            "Mnu",
            1,
            np.float32,
            "Msun",
            "Total neutrino particle mass within a sphere {label}",
        ),
    ]

    def __init__(self, cellgrid, SOval, type="mean"):
        super().__init__(cellgrid)

        if not type in ["mean", "crit", "physical", "BN98"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type

        # in the neutrino model, the mean neutrino density is implicitly
        # assumed to be based on Omega_nu_0 and critical_density_0
        # here, critical_density_0 = critical_density * (H0/H)**2
        # however, we need to scale this to the appropriate redshift,
        # hence an additional factor 1/a**3
        self.nu_density = (
            cellgrid.cosmology["Omega_nu_0"]
            * cellgrid.critical_density
            * (
                cellgrid.cosmology["H0 [internal units]"]
                / cellgrid.cosmology["H [internal units]"]
            )
            ** 2
            / cellgrid.a**3
        )

        # This specifies how large a sphere is read in:
        # we use default values that are sufficiently small/large to avoid reading in too many particles
        self.mean_density_multiple = 1000.0
        self.critical_density_multiple = 1000.0
        self.physical_radius_mpc = 0.0
        if type == "mean":
            self.mean_density_multiple = SOval
        elif type == "crit":
            self.critical_density_multiple = SOval
        elif type == "BN98":
            self.critical_density_multiple = cellgrid.virBN98
        elif type == "physical":
            self.physical_radius_mpc = 0.001 * SOval

        # Give this calculation a name so we can select it on the command line
        if type in ["mean", "crit"]:
            self.name = f"SO_{SOval:.0f}_{type}"
        elif type == "physical":
            self.name = f"SO_{SOval:.0f}_kpc"
        elif type == "BN98":
            self.name = "SO_BN98"

        # set some variables that are used during the calculation and that do not change
        if self.type == "crit":
            self.reference_density = (
                self.critical_density_multiple * self.critical_density
            )
            self.SO_name = f"{self.critical_density_multiple:.0f}_crit"
            self.label = f"within which the density is {self.critical_density_multiple:.0f} times the critical value"
        elif self.type == "mean":
            self.reference_density = self.mean_density_multiple * self.mean_density
            self.SO_name = f"{self.mean_density_multiple:.0f}_mean"
            self.label = f"within which the density is {self.mean_density_multiple:.0f} times the mean value"
        elif self.type == "physical":
            self.reference_density = 0.0
            self.SO_name = f"{1000. * self.physical_radius_mpc:.0f}_kpc"
            self.label = f"with a radius of {1000. * self.physical_radius_mpc:.0f} kpc"
        elif self.type == "BN98":
            self.reference_density = (
                self.critical_density_multiple * self.critical_density
            )
            self.SO_name = "BN98"
            self.label = f"within which the density is {self.critical_density_multiple:.2f} times the critical value"

    def calculate(self, input_halo, search_radius, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        search_radius    - radius in which we have all particles
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Find the halo centre of potential
        centre = input_halo["cofp"]

        (nr_parts, ordered_radius, density, cumulative_mass, radius, types, is_bound_to_satellite,
         position, velocity, mass) = cumulative_mass_and_density(input_halo, self.nu_density, data)

        reg = mass.units.registry

        SO = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, shape, dtype, unit, _ in self.SO_properties:
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            SO[name] = unyt.unyt_array(val, dtype=dtype, units=unit, registry=reg)

        # Check if we ever reach the density threshold
        if self.reference_density > 0.0 * self.reference_density:
            if nr_parts > 0:
                try:
                    SO_r, SO_mass, SO_volume = find_SO_radius_and_mass(
                        ordered_radius,
                        density,
                        cumulative_mass,
                        self.reference_density,
                    )
                    SO["r"] += SO_r
                    SO["mass"] += SO_mass
                except ReadRadiusTooSmallError:
                    raise ReadRadiusTooSmallError("SO radius multiple was too small!")
            else:
                SO_volume = 4.0 * np.pi / 3.0 * SO["r"] ** 3
        elif self.physical_radius_mpc > 0.0:
            SO["r"] += self.physical_radius_mpc * unyt.Mpc
            SO_volume = 4.0 * np.pi / 3.0 * SO["r"] ** 3
            if nr_parts > 0:
                # find the enclosed mass using interpolation
                i = np.argmax(ordered_radius > SO["r"])
                if i == 0:
                    # we only have particles in the centre, so we cannot interpolate
                    SO["mass"] += cumulative_mass[i]
                else:
                    r1 = ordered_radius[i - 1]
                    r2 = ordered_radius[i]
                    M1 = cumulative_mass[i - 1]
                    M2 = cumulative_mass[i]
                    SO["mass"] += M1 + (SO["r"] - r1) / (r2 - r1) * (M2 - M1)

        else:
            # if we get here, we must be in the case where physical_radius_mpc is supposed to be 0
            # that can only happen if we are looking at a multiple of some radius
            # in that case, SO["r"] should remain 0
            # in any other case, something went wrong
            if not hasattr(self, "multiple"):
                raise RuntimeError(
                    "Physical radius was set to 0! This should not happen!"
                )

        # the second condition is necessary to deal with physical SO radii and
        # no particles
        if SO["r"] > 0.0 * radius.units and SO["mass"] > 0.0 * mass.units:

            gas_selection = radius[types == "PartType0"] < SO["r"]
            dm_selection = radius[types == "PartType1"] < SO["r"]
            star_selection = radius[types == "PartType4"] < SO["r"]
            bh_selection = radius[types == "PartType5"] < SO["r"]

            all_selection = radius < SO["r"]
            mass = mass[all_selection]
            position = position[all_selection]
            velocity = velocity[all_selection]
            types = types[all_selection]
            is_bound_to_satellite = is_bound_to_satellite[all_selection]

            # we have to set the value like this to avoid conversion to dtype=np.float32...
            SO["N"] = unyt.unyt_array(
                all_selection.sum(),
                dtype=np.uint32,
                units="dimensionless",
                registry=reg,
            )

            # note that we cannot divide by mSO here, since that was based on an interpolation
            mass_frac = mass / mass.sum()
            SO["com"][:] = (mass_frac[:, None] * position).sum(axis=0)
            SO["com"][:] += centre
            SO["vcom"][:] = (mass_frac[:, None] * velocity).sum(axis=0)

            SO["Mfrac_satellites"] += mass[is_bound_to_satellite].sum() / SO["mass"]

            gas_masses = mass[types == "PartType0"]
            gas_pos = position[types == "PartType0"]
            gas_relpos = gas_pos[:, :] - SO["com"][None, :]
            gas_vel = velocity[types == "PartType0"]
            gas_relvel = gas_vel[:, :] - SO["vcom"][None, :]
            SO["mass_gas"] += gas_masses.sum()
            SO["Jgas"][:] = (
                gas_masses[:, None] * unyt.array.ucross(gas_relpos, gas_relvel)
            ).sum(axis=0)

            dm_masses = mass[types == "PartType1"]
            dm_relpos = position[types == "PartType1"][:, :] - SO["com"][None, :]
            dm_vel = velocity[types == "PartType1"]
            dm_relvel = dm_vel[:, :] - SO["vcom"][None, :]
            SO["mass_dm"] += dm_masses.sum()
            SO["JDM"][:] = (
                dm_masses[:, None] * unyt.array.ucross(dm_relpos, dm_relvel)
            ).sum(axis=0)

            star_masses = mass[types == "PartType4"]
            star_pos = position[types == "PartType4"]
            star_relpos = star_pos[:, :] - SO["com"][None, :]
            star_vel = velocity[types == "PartType4"]
            star_relvel = star_vel[:, :] - SO["vcom"][None, :]
            SO["mass_star"] += star_masses.sum()
            SO["Jstar"][:] = (
                star_masses[:, None] * unyt.array.ucross(star_relpos, star_relvel)
            ).sum(axis=0)

            SO["MBHdyn"] += mass[types == "PartType5"].sum()

            # gas specific properties. We (can) only do these if we have gas.
            # (remember that "PartType0" might not be part of 'data' at all)
            if np.any(gas_selection):
                SO["com_gas"][:] = (gas_masses[:, None] * gas_pos).sum(
                    axis=0
                ) / gas_masses.sum()
                SO["com_gas"][:] += centre
                SO["vcom_gas"][:] = (gas_masses[:, None] * gas_vel).sum(
                    axis=0
                ) / gas_masses.sum()
                vrel = gas_vel - SO["vcom"][None, :]
                SO["veldisp_gas"][0] += (
                    gas_masses * vrel[:, 0] * vrel[:, 0]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][1] += (
                    gas_masses * vrel[:, 1] * vrel[:, 1]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][2] += (
                    gas_masses * vrel[:, 2] * vrel[:, 2]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][3] += (
                    gas_masses * vrel[:, 0] * vrel[:, 1]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][4] += (
                    gas_masses * vrel[:, 0] * vrel[:, 2]
                ).sum() / gas_masses.sum()
                SO["veldisp_gas"][5] += (
                    gas_masses * vrel[:, 1] * vrel[:, 2]
                ).sum() / gas_masses.sum()

                SO["Mgasmetal"] += (
                    gas_masses * data["PartType0"]["MetalMassFractions"][gas_selection]
                ).sum()

                gas_temperatures = data["PartType0"]["Temperatures"][gas_selection]
                Tgas_selection = gas_temperatures > 1.0e5 * unyt.K
                SO["Mhotgas"] += gas_masses[Tgas_selection].sum()

                if np.any(Tgas_selection):
                    SO["Tgas"] += (
                        gas_temperatures[Tgas_selection] * gas_masses[Tgas_selection]
                    ).sum() / SO["Mhotgas"]

                SO["Xraylum"] += data["PartType0"]["XrayLuminosities"][
                    gas_selection
                ].sum()
                SO["Xrayphlum"] += data["PartType0"]["XrayPhotonLuminosities"][
                    gas_selection
                ].sum()

                # convert to 64-bit before calculating the sum to avoid overflow
                compY = data["PartType0"]["ComptonYParameters"][gas_selection].sum()
                # unyt has some internal issue that causes an overflow when
                # converting from compY.units to SO["compY"].units.
                # we avoid this issue by manually converting the unit
                unit = 1.0 * compY.units
                new_unit = unit.to(SO["compY"].units)
                SO["compY"] += compY.value * new_unit

                # below we need to force conversion to np.float64 before summing up particles
                # to avoid overflow
                ekin_gas = gas_masses * (
                    (velocity[types == "PartType0"] - SO["vcom"][None, :]) ** 2
                ).sum(axis=1)
                ekin_gas = unyt.unyt_array(
                    ekin_gas.value, dtype=np.float64, units=ekin_gas.units
                )
                SO["Ekin_gas"] += 0.5 * ekin_gas.sum()
                etherm_gas = (
                    1.5
                    * gas_masses
                    * data["PartType0"]["Pressures"][gas_selection]
                    / data["PartType0"]["Densities"][gas_selection]
                )
                etherm_gas = unyt.unyt_array(
                    etherm_gas.value, dtype=np.float64, units=etherm_gas.units
                )
                SO["Etherm_gas"] += etherm_gas.sum()

            # DM specific properties
            if np.any(dm_selection):
                vrel = dm_vel - SO["vcom"][None, :]
                SO["veldisp_dm"][0] += (
                    dm_masses * vrel[:, 0] * vrel[:, 0]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][1] += (
                    dm_masses * vrel[:, 1] * vrel[:, 1]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][2] += (
                    dm_masses * vrel[:, 2] * vrel[:, 2]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][3] += (
                    dm_masses * vrel[:, 0] * vrel[:, 1]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][4] += (
                    dm_masses * vrel[:, 0] * vrel[:, 2]
                ).sum() / dm_masses.sum()
                SO["veldisp_dm"][5] += (
                    dm_masses * vrel[:, 1] * vrel[:, 2]
                ).sum() / dm_masses.sum()

            # star specific properties
            if np.any(star_selection):
                SO["com_star"][:] = (star_masses[:, None] * star_pos).sum(
                    axis=0
                ) / star_masses.sum()
                SO["com_star"][:] += centre
                SO["vcom_star"][:] = (star_masses[:, None] * star_vel).sum(
                    axis=0
                ) / star_masses.sum()
                vrel = star_vel - SO["vcom"][None, :]
                SO["veldisp_star"][0] += (
                    star_masses * vrel[:, 0] * vrel[:, 0]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][1] += (
                    star_masses * vrel[:, 1] * vrel[:, 1]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][2] += (
                    star_masses * vrel[:, 2] * vrel[:, 2]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][3] += (
                    star_masses * vrel[:, 0] * vrel[:, 1]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][4] += (
                    star_masses * vrel[:, 0] * vrel[:, 2]
                ).sum() / star_masses.sum()
                SO["veldisp_star"][5] += (
                    star_masses * vrel[:, 1] * vrel[:, 2]
                ).sum() / star_masses.sum()

                SO["Mstarinit"] += data["PartType4"]["InitialMasses"][
                    star_selection
                ].sum()
                SO["Mstarmetal"] += (
                    star_masses
                    * data["PartType4"]["MetalMassFractions"][star_selection]
                ).sum()
                SO["Lstar"][:] = data["PartType4"]["Luminosities"][star_selection].sum()

            # BH specific properties
            if np.any(bh_selection):
                SO["MBHsub"] += data["PartType5"]["SubgridMasses"][bh_selection].sum()
                agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][
                    bh_selection
                ]

                SO["BHlasteventa"] += np.max(agn_eventa)

                iBHmax = np.argmax(data["PartType5"]["SubgridMasses"][bh_selection])
                SO["BHmaxM"] += data["PartType5"]["SubgridMasses"][bh_selection][iBHmax]
                # unyt annoyingly converts to a floating point type if you use '+='
                # the only way to avoid this is by directly setting the data for the unyt_array
                # however, that is unsafe and results in a warning
                # the only option left is to replace the array with a new copy
                SO["BHmaxID"] = unyt.unyt_array(
                    data["PartType5"]["ParticleIDs"][bh_selection][iBHmax].value,
                    dtype=SO["BHmaxID"].dtype,
                    units=SO["BHmaxID"].units,
                )
                SO["BHmaxpos"] += data["PartType5"]["Coordinates"][bh_selection][iBHmax]
                SO["BHmaxvel"] += data["PartType5"]["Velocities"][bh_selection][iBHmax]
                SO["BHmaxAR"] += data["PartType5"]["AccretionRates"][bh_selection][
                    iBHmax
                ]
                SO["BHmaxlasteventa"] += agn_eventa[iBHmax]

            # Neutrino specific properties
            if "PartType6" in data:
                pos = data["PartType6"]["Coordinates"] - centre[None, :]
                nur = np.sqrt(np.sum(pos**2, axis=1))
                nu_selection = nur < SO["r"]
                SO["Mnu"] += data["PartType6"]["Masses"][nu_selection].sum()
                SO["MnuNS"] += (
                    data["PartType6"]["Masses"][nu_selection]
                    * data["PartType6"]["Weights"][nu_selection]
                ).sum()
                SO["MnuNS"] += self.nu_density * SO_volume

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        for name, _, _, _, description in self.SO_properties:
            halo_result.update(
                {
                    f"SO/{self.SO_name}/{name}": (
                        SO[name],
                        description.format(label=self.label),
                    )
                }
            )

        return


class RadiusMultipleSOProperties(SOProperties):
    def __init__(self, cellgrid, SOval, multiple, type="mean"):
        if not type in ["mean", "crit"]:
            raise AttributeError(
                "SOs with a radius that is a multiple of another SO radius are only allowed for type mean or crit!"
            )

        # initialise the SOProperties object using a conservative physical radius estimate
        super().__init__(cellgrid, 3000.0, "physical")

        # overwrite the name, SO_name and label
        self.SO_name = f"{multiple:.0f}xR_{SOval:.0f}_{type}"
        self.label = f"with a radius that is {self.SO_name}"
        self.name = f"SO_{self.SO_name}"

        self.requested_type = type
        self.requested_SOval = SOval
        self.multiple = multiple

    def calculate(self, input_halo, search_radius, data, halo_result):

        # find the actual physical radius we want
        key = f"SO/{self.requested_SOval:.0f}_{self.requested_type}/r"
        if not key in halo_result:
            raise RuntimeError(
                f"Trying to obtain {key}, but the corresponding SO radius has not been calculated!"
            )
        self.physical_radius_mpc = self.multiple * (halo_result[key][0].to("Mpc").value)

        # Check that we read in a large enough radius
        if self.multiple * halo_result[key][0] > search_radius:
            raise ReadRadiusTooSmallError("SO radius multiple estimate was too small!")

        super().calculate(input_halo, search_radius, data, halo_result)
        return


def test_SO_properties():

    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(4251)
    property_calculator_50kpc = SOProperties(
        dummy_halos.get_cell_grid(), 50.0, "physical"
    )
    property_calculator_2500mean = SOProperties(
        dummy_halos.get_cell_grid(), 2500.0, "mean"
    )
    property_calculator_2500crit = SOProperties(
        dummy_halos.get_cell_grid(), 2500.0, "crit"
    )
    property_calculator_BN98 = SOProperties(dummy_halos.get_cell_grid(), 0.0, "BN98")
    property_calculator_5x2500mean = RadiusMultipleSOProperties(
        dummy_halos.get_cell_grid(), 2500.0, 5.0, "mean"
    )

    for i in range(100):
        input_halo, data, rmax, Mtot, Npart = dummy_halos.get_random_halo(
            [2, 10, 100, 1000, 10000], has_neutrinos=True
        )
        rho_ref = Mtot / (4.0 / 3.0 * np.pi * rmax**3)

        # force the SO radius to be outside the search sphere and check that
        # we get a ReadRadiusTooSmallError
        property_calculator_2500mean.reference_density = 0.01 * rho_ref
        property_calculator_2500crit.reference_density = 0.01 * rho_ref
        property_calculator_BN98.reference_density = 0.01 * rho_ref
        for prop_calc in [
            property_calculator_2500mean,
            property_calculator_2500crit,
            property_calculator_BN98,
        ]:
            fail = False
            try:
                halo_result = {}
                prop_calc.calculate(input_halo, rmax, data, halo_result)
            except ReadRadiusTooSmallError:
                fail = True
            # 1 particle halos don't fail, since we always assume that the first
            # particle is at the centre of potential (which means we exclude it
            # in the SO calculation)
            assert (Npart == 1) or fail

        # force the radius multiple to trip over not having computed the
        # required radius
        fail = False
        try:
            halo_result = {}
            property_calculator_5x2500mean.calculate(
                input_halo, rmax, data, halo_result
            )
        except RuntimeError:
            fail = True
        assert fail

        # force the radius multiple to trip over the search radius
        fail = False
        try:
            halo_result = {"SO/2500_mean/r": (0.1 * rmax, "Dummy value.")}
            property_calculator_5x2500mean.calculate(
                input_halo, 0.2 * rmax, data, halo_result
            )
        except ReadRadiusTooSmallError:
            fail = True
        assert fail

        # force the SO radius to be within the search sphere
        property_calculator_2500mean.reference_density = 2.0 * rho_ref
        property_calculator_2500crit.reference_density = 2.0 * rho_ref
        property_calculator_BN98.reference_density = 2.0 * rho_ref

        for SO_name, prop_calc in [
            ("50_kpc", property_calculator_50kpc),
            ("2500_mean", property_calculator_2500mean),
            ("2500_crit", property_calculator_2500crit),
            ("BN98", property_calculator_BN98),
            ("5xR_2500_mean", property_calculator_5x2500mean),
        ]:

            halo_result = {}
            # make sure the radius multiple is found this time
            if SO_name == "5xR_2500_mean":
                halo_result["SO/2500_mean/r"] = (
                    0.1 * rmax,
                    "Dummy value to force correct behaviour",
                )
            input_data = {}
            for ptype in prop_calc.particle_properties:
                if ptype in data:
                    input_data[ptype] = {}
                    for dset in prop_calc.particle_properties[ptype]:
                        input_data[ptype][dset] = data[ptype][dset]
            input_halo_copy = input_halo.copy()
            input_data_copy = input_data.copy()
            prop_calc.calculate(input_halo, rmax, input_data, halo_result)
            # make sure the calculation does not change the input
            assert input_halo_copy == input_halo
            assert input_data_copy == input_data

            for name, size, dtype, unit_string, _ in prop_calc.SO_properties:
                full_name = f"SO/{SO_name}/{name}"
                assert full_name in halo_result
                result = halo_result[full_name][0]
                assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
                assert result.dtype == dtype
                unit = unyt.Unit(unit_string)
                assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone mode. Just run test_SO_properties().
    """
    print("Calling test_SO_properties()...")
    test_SO_properties()
    print("Test passed.")

    print("Name & Size & Unit & Type & Description \\\\")
    for (
        name,
        size,
        dtype,
        unit,
        description,
    ) in SOProperties.SO_properties:
        unit_str = unit.__str__()
        unit_str = unit_str.replace("1.98841586e+30 kg", "M$_\\odot{}$")
        unit_str = unit_str.replace(">", "$>$")
        print(
            f"\\verb+{name}+ & {size} & {unit_str} & {dtype.__name__} & {description.format(label='with a radius as determined for this particular SO type.')} \\\\"
        )
