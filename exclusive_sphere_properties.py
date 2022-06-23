#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius
from kinematic_properties import (
    get_velocity_dispersion_matrix,
    get_angular_momentum,
    get_angular_momentum_and_kappa_corot,
)

from astropy.cosmology import w0waCDM, z_at_value
import astropy.constants as const
import astropy.units as astropy_units

# index of elements O and Fe in the SmoothedElementMassFractions dataset
indexO = 4
indexFe = 8


class RecentlyHeatedGasFilter:
    """
    Filter used to determine whether gas particles should be considered to be
    "recently heated".

    This corresponds to the lightcone map filter used in SWIFT itself, which
    filters out gas particles for which LastAGNFeedbackScaleFactors is less
    than 15 Myr ago, and within some temperature bracket.

    Since the conversion from a time difference to a scale factor is not
    trivial, we compute the corresponding scale factor limit only once using
    the correct astropy.cosmology.
    """

    def __init__(
        self,
        cellgrid,
        delta_time=15.0 * unyt.Myr,
        delta_logT_min=-1.0,
        delta_logT_max=0.3,
        AGN_delta_T=8.80144197177e7 * unyt.K,
    ):
        H0 = unyt.unyt_quantity(
            cellgrid.cosmology["H0 [internal units]"],
            units="1/snap_time",
            registry=cellgrid.snap_unit_registry,
        ).to("1/s")

        Omega_b = cellgrid.cosmology["Omega_b"]
        Omega_lambda = cellgrid.cosmology["Omega_lambda"]
        Omega_r = cellgrid.cosmology["Omega_r"]
        Omega_m = cellgrid.cosmology["Omega_m"]
        w_0 = cellgrid.cosmology["w_0"]
        w_a = cellgrid.cosmology["w_a"]
        z_now = cellgrid.cosmology["Redshift"]

        # expressions taken directly from astropy, since they do no longer
        # allow access to these attributes (since version 5.1+)
        critdens_const = (3.0 / (8.0 * np.pi * const.G)).cgs.value
        a_B_c2 = (4.0 * const.sigma_sb / const.c**3).cgs.value

        # SWIFT provides Omega_r, but we need a consistent Tcmb0 for astropy.
        # This is an exact inversion of the procedure performed in astropy.
        critical_density_0 = astropy_units.Quantity(
            critdens_const * H0.to("1/s").value ** 2,
            astropy_units.g / astropy_units.cm**3,
        )

        Tcmb0 = (Omega_r * critical_density_0.value / a_B_c2) ** (1.0 / 4.0)

        cosmology = w0waCDM(
            H0=H0.to_astropy(),
            Om0=Omega_m,
            Ode0=Omega_lambda,
            w0=w_0,
            wa=w_a,
            Tcmb0=Tcmb0,
            Ob0=Omega_b,
        )

        lookback_time_now = cosmology.lookback_time(z_now)
        lookback_time_limit = lookback_time_now + delta_time.to_astropy()
        z_limit = z_at_value(cosmology.lookback_time, lookback_time_limit)

        # for some reason, the return type of z_at_value has changed between
        # astropy versions. We make sure it is not some astropy quantity
        # before using it.
        if hasattr(z_limit, "value"):
            z_limit = z_limit.value

        self.a_limit = 1.0 / (1.0 + z_limit) * unyt.dimensionless

        self.Tmin = AGN_delta_T * 10.0**delta_logT_min
        self.Tmax = AGN_delta_T * 10.0**delta_logT_max

    def is_recently_heated(self, lastAGNfeedback, temperature):
        return (
            (lastAGNfeedback >= self.a_limit)
            & (temperature >= self.Tmin)
            & (temperature <= self.Tmax)
        )


class ExclusiveSphereProperties(HaloProperty):
    """
    Compute exclusive sphere properties for halos.

    The exclusive sphere has a fixed radius and only includes particles that
    are bound to the halo.
    """

    # List of particle properties we need to read in
    particle_properties = {
        "PartType0": [
            "Coordinates",
            "GroupNr_bound",
            "LastAGNFeedbackScaleFactors",
            "Masses",
            "MetalMassFractions",
            "SmoothedElementMassFractions",
            "StarFormationRates",
            "Temperatures",
            "Velocities",
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
    }

    # List of properties that get computed
    # For each property, we have the following columns:
    #  - name: Name of the property within calculate() and in the output file
    #  - shape: Shape of this property for a single halo (1: scalar, 3: vector...)
    #  - dtype: Data type that will be used. Should have enough precision to avoid over/underflow
    #  - unit: Units that will be used internally and for the output.
    #  - description: Description string that will be used to describe the property in the output.
    exclusive_sphere_properties = [
        ("Mtot", 1, np.float32, unyt.Msun, "Total mass"),
        ("Mgas", 1, np.float32, unyt.Msun, "Total gas mass"),
        ("Mdm", 1, np.float32, unyt.Msun, "Total DM mass"),
        ("Mstar", 1, np.float32, unyt.Msun, "Total stellar mass"),
        ("Mstar_init", 1, np.float32, unyt.Msun, "Total initial stellar mass"),
        ("Mbh", 1, np.float32, unyt.Msun, "Total BH dynamical mass"),
        ("Mbh_subgrid", 1, np.float32, unyt.Msun, "Total BH subgrid mass"),
        ("Ngas", 1, np.uint32, unyt.dimensionless, "Number of gas particles."),
        ("Ndm", 1, np.uint32, unyt.dimensionless, "Number of dark matter particles."),
        ("Nstar", 1, np.uint32, unyt.dimensionless, "Number of star particles."),
        ("Nbh", 1, np.uint32, unyt.dimensionless, "Number of black hole particles."),
        (
            "BHlasteventa",
            1,
            np.float32,
            unyt.dimensionless,
            "Scale-factor of last AGN event.",
        ),
        ("BHmaxM", 1, np.float32, unyt.Msun, "Mass of most massive black hole."),
        ("BHmaxID", 1, np.uint64, unyt.dimensionless, "ID of most massive black hole."),
        ("BHmaxpos", 3, np.float64, unyt.kpc, "Position of most massive black hole."),
        (
            "BHmaxvel",
            3,
            np.float32,
            unyt.km / unyt.s,
            "Velocity of most massive black hole.",
        ),
        (
            "BHmaxAR",
            1,
            np.float32,
            unyt.Msun / unyt.yr,
            "Accretion rate of most massive black hole.",
        ),
        (
            "BHmaxlasteventa",
            1,
            np.float32,
            unyt.dimensionless,
            "Scale-factor of last AGN event for most massive black hole.",
        ),
        ("com", 3, np.float32, unyt.kpc, "Centre of mass"),
        ("vcom", 3, np.float32, unyt.km / unyt.s, "Centre of mass velocity"),
        (
            "Lgas",
            3,
            np.float32,
            unyt.Msun * unyt.kpc * unyt.km / unyt.s,
            "Total angular momentum of the gas, relative w.r.t. the gas centre of mass.",
        ),
        (
            "Ldm",
            3,
            np.float32,
            unyt.Msun * unyt.kpc * unyt.km / unyt.s,
            "Total angular momentum of the dark matter, relative w.r.t. the dark matter centre of mass.",
        ),
        (
            "Lstar",
            3,
            np.float32,
            unyt.Msun * unyt.kpc * unyt.km / unyt.s,
            "Total angular momentum of the stars, relative w.r.t. the stellar centre of mass.",
        ),
        ("kappa_corot_gas", 1, np.float32, unyt.dimensionless, "Kappa corot for gas."),
        (
            "kappa_corot_star",
            1,
            np.float32,
            unyt.dimensionless,
            "Kappa corot for stars.",
        ),
        (
            "veldisp_gas",
            6,
            np.float32,
            unyt.km**2 / unyt.s**2,
            "Velocity dispersion of the gas. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        (
            "veldisp_dm",
            6,
            np.float32,
            unyt.km**2 / unyt.s**2,
            "Velocity dispersion of the dark matter. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        (
            "veldisp_star",
            6,
            np.float32,
            unyt.km**2 / unyt.s**2,
            "Velocity dispersion of the stars. Measured relative to the centre of mass velocity of all particles. The order of the components of the dispersion tensor is XX YY ZZ XY XZ YZ.",
        ),
        ("Mgas_SF", 1, np.float32, unyt.Msun, "Total mass of star-forming gas."),
        ("Mgas_noSF", 1, np.float32, unyt.Msun, "Total mass of non star-forming gas."),
        ("Mgasmetal", 1, np.float32, unyt.Msun, "Total gas mass in metals."),
        (
            "Mgasmetal_SF",
            1,
            np.float32,
            unyt.Msun,
            "Total gas mass in metals for gas that is star-forming.",
        ),
        (
            "Mgasmetal_noSF",
            1,
            np.float32,
            unyt.Msun,
            "Total gas mass in metals for gas that is non star-forming.",
        ),
        ("MgasO", 1, np.float32, unyt.Msun, "Total gas mass in oxygen."),
        (
            "MgasO_SF",
            1,
            np.float32,
            unyt.Msun,
            "Total gas mass in oxygen for gas that is star-forming.",
        ),
        (
            "MgasO_noSF",
            1,
            np.float32,
            unyt.Msun,
            "Total gas mass in oxygen for gas that is non star-forming.",
        ),
        ("MgasFe", 1, np.float32, unyt.Msun, "Total gas mass in iron."),
        (
            "MgasFe_SF",
            1,
            np.float32,
            unyt.Msun,
            "Total gas mass in iron for gas that is star-forming.",
        ),
        (
            "MgasFe_noSF",
            1,
            np.float32,
            unyt.Msun,
            "Total gas mass in iron for gas that is non star-forming.",
        ),
        ("Tgas", 1, np.float32, unyt.K, "Mass-weighted gas temperature."),
        (
            "Tgas_no_agn",
            1,
            np.float32,
            unyt.K,
            "Mass-weighted gas temperature, excluding gas that was recently heated by AGN.",
        ),
        ("SFR", 1, np.float32, unyt.Msun / unyt.yr, "Total SFR"),
        ("Luminosity", 9, np.float32, unyt.dimensionless, "Total luminosity"),
        ("Mstarmetal", 1, np.float32, unyt.Msun, "Total stellar mass in metals."),
        ("HalfMassRadiusTot", 1, np.float32, unyt.kpc, "Total half mass radius"),
        ("HalfMassRadiusGas", 1, np.float32, unyt.kpc, "Total gas half mass radius"),
        ("HalfMassRadiusDM", 1, np.float32, unyt.kpc, "Total DM half mass radius"),
        (
            "HalfMassRadiusStar",
            1,
            np.float32,
            unyt.kpc,
            "Total stellar half mass radius",
        ),
    ]

    def __init__(self, cellgrid, physical_radius_kpc, recently_heated_gas_filter):
        """
        Construct an ExclusiveSphereProperties object with the given physical
        radius (in Mpc) that uses the given filter to filter out recently
        heated gas particles.
        """

        super().__init__(cellgrid)

        self.filter = recently_heated_gas_filter

        # no density criterion for these properties
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"exclusive_sphere_{physical_radius_kpc:.0f}kpc"

    def calculate(self, input_halo, search_radius, data, halo_result, cache=None):
        """
        Compute centre of mass etc of bound particles

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        centre = input_halo["cofp"]
        index = input_halo["index"]

        types_present = [type for type in self.particle_properties if type in data]

        mass = []
        position = []
        radius = []
        velocity = []
        types = []
        for ptype in types_present:
            grnr = data[ptype]["GroupNr_bound"]
            in_halo = grnr == index
            mass.append(data[ptype][mass_dataset(ptype)][in_halo])
            pos = data[ptype]["Coordinates"][in_halo, :] - centre[None, :]
            position.append(pos)
            r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
            radius.append(r)
            velocity.append(data[ptype]["Velocities"][in_halo, :])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        mass = unyt.array.uconcatenate(mass)
        position = unyt.array.uconcatenate(position)
        radius = unyt.array.uconcatenate(radius)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)

        exclusive_sphere = {}
        # declare all the variables we will compute
        # we set them to 0 in case a particular variable cannot be computed
        # all variables are defined with physical units and an appropriate dtype
        # we need to use the custom unit registry so that everything can be converted
        # back to snapshot units in the end
        for name, shape, dtype, unit, _ in self.exclusive_sphere_properties:
            if shape > 1:
                val = [0] * shape
            else:
                val = 0
            exclusive_sphere[name] = unyt.unyt_array(
                val, dtype=dtype, units=unit, registry=mass.units.registry
            )

        mask = radius <= self.physical_radius_mpc * unyt.Mpc

        mass = mass[mask]
        position = position[mask]
        velocity = velocity[mask]
        radius = radius[mask]
        type = types[mask]

        gas_mask_ap = mask[types == "PartType0"]
        dm_mask_ap = mask[types == "PartType1"]
        star_mask_ap = mask[types == "PartType4"]
        bh_mask_ap = mask[types == "PartType5"]

        exclusive_sphere["Ngas"] = (
            gas_mask_ap.sum(dtype=exclusive_sphere["Ngas"].dtype)
            * exclusive_sphere["Ngas"].units
        )
        exclusive_sphere["Ndm"] = (
            dm_mask_ap.sum(dtype=exclusive_sphere["Ndm"].dtype)
            * exclusive_sphere["Ndm"].units
        )
        exclusive_sphere["Nstar"] = (
            star_mask_ap.sum(dtype=exclusive_sphere["Nstar"].dtype)
            * exclusive_sphere["Nstar"].units
        )
        exclusive_sphere["Nbh"] = (
            bh_mask_ap.sum(dtype=exclusive_sphere["Nbh"].dtype)
            * exclusive_sphere["Nbh"].units
        )

        mass_gas = mass[type == "PartType0"]
        mass_dm = mass[type == "PartType1"]
        mass_star = mass[type == "PartType4"]

        pos_gas = position[type == "PartType0"]
        pos_dm = position[type == "PartType1"]
        pos_star = position[type == "PartType4"]

        vel_gas = velocity[type == "PartType0"]
        vel_dm = velocity[type == "PartType1"]
        vel_star = velocity[type == "PartType4"]

        exclusive_sphere["Mtot"] += mass.sum()
        exclusive_sphere["Mgas"] += mass_gas.sum()
        exclusive_sphere["Mdm"] = mass_dm.sum()
        exclusive_sphere["Mstar"] += mass_star.sum()
        if exclusive_sphere["Nstar"] > 0:
            star_mask_all = data["PartType4"]["GroupNr_bound"] == index
            exclusive_sphere["Mstar_init"] += data["PartType4"]["InitialMasses"][
                star_mask_all
            ][star_mask_ap].sum()
            exclusive_sphere["Luminosity"] += data["PartType4"]["Luminosities"][
                star_mask_all
            ][star_mask_ap].sum(axis=0)
            exclusive_sphere["Mstarmetal"] += (
                mass_star
                * data["PartType4"]["MetalMassFractions"][star_mask_all][star_mask_ap]
            ).sum()
        exclusive_sphere["Mbh"] = mass[type == "PartType5"].sum()
        if exclusive_sphere["Nbh"] > 0:
            bh_mask_all = data["PartType5"]["GroupNr_bound"] == index
            exclusive_sphere["Mbh_subgrid"] += data["PartType5"]["SubgridMasses"][
                bh_mask_all
            ][bh_mask_ap].sum()

            agn_eventa = data["PartType5"]["LastAGNFeedbackScaleFactors"][bh_mask_all][
                bh_mask_ap
            ]

            exclusive_sphere["BHlasteventa"] += np.max(agn_eventa)

            iBHmax = np.argmax(
                data["PartType5"]["SubgridMasses"][bh_mask_all][bh_mask_ap]
            )
            exclusive_sphere["BHmaxM"] += data["PartType5"]["SubgridMasses"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxID"] = (
                data["PartType5"]["ParticleIDs"][bh_mask_all][bh_mask_ap][
                    iBHmax
                ].astype(exclusive_sphere["BHmaxID"].dtype)
                * exclusive_sphere["BHmaxID"].units
            )
            exclusive_sphere["BHmaxpos"][:] = data["PartType5"]["Coordinates"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxvel"][:] = data["PartType5"]["Velocities"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxAR"] += data["PartType5"]["AccretionRates"][
                bh_mask_all
            ][bh_mask_ap][iBHmax]
            exclusive_sphere["BHmaxlasteventa"] += agn_eventa[iBHmax]

        if exclusive_sphere["Mtot"] > 0.0 * exclusive_sphere["Mtot"].units:
            mfrac = mass / exclusive_sphere["Mtot"]
            exclusive_sphere["com"][:] = (mfrac[:, None] * position).sum(axis=0)
            exclusive_sphere["com"][:] += centre
            exclusive_sphere["vcom"][:] = (mfrac[:, None] * velocity).sum(axis=0)

        if exclusive_sphere["Mgas"] > 0.0 * exclusive_sphere["Mgas"].units:
            frac_mgas = mass_gas / exclusive_sphere["Mgas"]
            com_gas = (frac_mgas[:, None] * pos_gas).sum(axis=0)
            vcom_gas = (frac_mgas[:, None] * vel_gas).sum(axis=0)
            Lgas, kappa = get_angular_momentum_and_kappa_corot(
                mass_gas, pos_gas, vel_gas, com_gas, vcom_gas
            )
            exclusive_sphere["Lgas"][:] = Lgas
            exclusive_sphere["kappa_corot_gas"] += kappa

            exclusive_sphere["veldisp_gas"][:] = get_velocity_dispersion_matrix(
                frac_mgas, vel_gas, exclusive_sphere["vcom"]
            )

        if exclusive_sphere["Mdm"] > 0.0 * exclusive_sphere["Mdm"].units:
            frac_mdm = mass_dm / exclusive_sphere["Mdm"]
            com_dm = (frac_mdm[:, None] * pos_dm).sum(axis=0)
            vcom_dm = (frac_mdm[:, None] * vel_dm).sum(axis=0)
            exclusive_sphere["Ldm"][:] = get_angular_momentum(
                mass_dm, pos_dm, vel_dm, com_dm, vcom_dm
            )

            exclusive_sphere["veldisp_dm"][:] = get_velocity_dispersion_matrix(
                frac_mdm, vel_dm, exclusive_sphere["vcom"]
            )

        if exclusive_sphere["Mstar"] > 0.0 * exclusive_sphere["Mstar"].units:
            frac_mstar = mass_star / exclusive_sphere["Mstar"]
            com_star = (frac_mstar[:, None] * pos_star).sum(axis=0)
            vcom_star = (frac_mstar[:, None] * vel_star).sum(axis=0)
            Lstar, kappa = get_angular_momentum_and_kappa_corot(
                mass_star, pos_star, vel_star, com_star, vcom_star
            )
            exclusive_sphere["Lstar"][:] = Lstar
            exclusive_sphere["kappa_corot_star"] += kappa

            exclusive_sphere["veldisp_star"][:] = get_velocity_dispersion_matrix(
                frac_mstar, vel_star, exclusive_sphere["vcom"]
            )

        if exclusive_sphere["Ngas"] > 0:
            gas_mask_all = data["PartType0"]["GroupNr_bound"] == index
            SFR = data["PartType0"]["StarFormationRates"][gas_mask_all][gas_mask_ap]
            # negative values of SFR are not SFR at all!
            is_SFR = SFR > 0.0
            exclusive_sphere["SFR"] += SFR[is_SFR].sum()
            exclusive_sphere["Mgas_SF"] += mass_gas[is_SFR].sum()
            exclusive_sphere["Mgas_noSF"] += mass_gas[~is_SFR].sum()
            Mgasmetal = (
                mass_gas
                * data["PartType0"]["MetalMassFractions"][gas_mask_all][gas_mask_ap]
            )
            exclusive_sphere["Mgasmetal_SF"] += Mgasmetal[is_SFR].sum()
            exclusive_sphere["Mgasmetal_noSF"] += Mgasmetal[~is_SFR].sum()
            exclusive_sphere["Mgasmetal"] += Mgasmetal.sum()
            MgasO = (
                mass_gas
                * data["PartType0"]["SmoothedElementMassFractions"][gas_mask_all][
                    gas_mask_ap
                ][:, indexO]
            )
            exclusive_sphere["MgasO_SF"] += MgasO[is_SFR].sum()
            exclusive_sphere["MgasO_noSF"] += MgasO[~is_SFR].sum()
            exclusive_sphere["MgasO"] += MgasO.sum()
            MgasFe = (
                mass_gas
                * data["PartType0"]["SmoothedElementMassFractions"][gas_mask_all][
                    gas_mask_ap
                ][:, indexFe]
            )
            exclusive_sphere["MgasFe_SF"] += MgasFe[is_SFR].sum()
            exclusive_sphere["MgasFe_noSF"] += MgasFe[~is_SFR].sum()
            exclusive_sphere["MgasFe"] += MgasFe.sum()
            gas_temp = data["PartType0"]["Temperatures"][gas_mask_all][gas_mask_ap]
            last_agn_gas = data["PartType0"]["LastAGNFeedbackScaleFactors"][
                gas_mask_all
            ][gas_mask_ap]
            no_agn = ~self.filter.is_recently_heated(last_agn_gas, gas_temp)
            exclusive_sphere["Tgas"] += (
                (mass_gas / exclusive_sphere["Mgas"]) * gas_temp
            ).sum()
            if np.any(no_agn):
                mass_gas_no_agn = mass_gas[no_agn]
                Mgas_no_agn = mass_gas_no_agn.sum()
                if Mgas_no_agn > 0.0:
                    exclusive_sphere["Tgas_no_agn"] += (
                        (mass_gas_no_agn / Mgas_no_agn) * gas_temp[no_agn]
                    ).sum()

        for name, r, m, M in zip(
            [
                "HalfMassRadiusTot",
                "HalfMassRadiusGas",
                "HalfMassRadiusDM",
                "HalfMassRadiusStar",
            ],
            [
                radius,
                radius[type == "PartType0"],
                radius[type == "PartType1"],
                radius[type == "PartType4"],
            ],
            [mass, mass_gas, mass_dm, mass_star],
            [
                exclusive_sphere["Mtot"],
                exclusive_sphere["Mgas"],
                exclusive_sphere["Mdm"],
                exclusive_sphere["Mstar"],
            ],
        ):
            exclusive_sphere[name] += get_half_mass_radius(r, m, M)
            if exclusive_sphere[name] >= self.physical_radius_mpc * unyt.Mpc:
                raise RuntimeError(
                    "Half mass radius '{name}' larger than aperture"
                    f" ({exclusive_sphere[name]} >="
                    f" {self.physical_radius_mpc * unyt.Mpc}!"
                    " This should not happen."
                )

        prefix = f"ExclusiveSphere/{self.physical_radius_mpc*1000.:.0f}kpc"
        for name, _, _, _, description in self.exclusive_sphere_properties:
            halo_result.update(
                {
                    f"{prefix}/{name}": (
                        exclusive_sphere[name],
                        description,
                    )
                }
            )

        return


class DummyExclusiveSphereProperties(ExclusiveSphereProperties):
    """
    Dummy ExclusiveSphereProperties object that can be used to test the code.

    The dummy object does not require any input arguments, except for a
    DummyHaloGenerator that can generate a minimal cellgrid that is needed
    for the RecentlyHeatedGasFilter.
    """

    def __init__(self, dummy_halos):

        self.filter = RecentlyHeatedGasFilter(dummy_halos.get_cell_grid())
        self.physical_radius_mpc = 0.05


def test_exclusive_sphere_properties():
    """
    Unit test for the exclusive sphere property calculations.

    We generate 100 random "dummy" halos and feed them to
    ExclusiveSphereProperties::calculate(). We check that the returned values
    are present, and have the right units, size and dtype
    """

    from dummy_halo_generator import DummyHaloGenerator

    # initialise the DummyHaloGenerator with a random seed
    dummy_halos = DummyHaloGenerator(3256)

    # generate a minimal ExclusiveSphereProperties object that does not require
    # an actual snapshot
    property_calculator = DummyExclusiveSphereProperties(dummy_halos)

    # generate 100 random halos
    for i in range(100):
        input_halo, data, _, _, _ = dummy_halos.get_random_halo(
            [1, 10, 100, 1000, 10000]
        )

        input_data = {}
        for ptype in property_calculator.particle_properties:
            if ptype in data:
                input_data[ptype] = {}
                for dset in property_calculator.particle_properties[ptype]:
                    input_data[ptype][dset] = data[ptype][dset]
        input_halo_copy = input_halo.copy()
        input_data_copy = input_data.copy()
        halo_result = {}
        property_calculator.calculate(
            input_halo, 0.0 * unyt.kpc, input_data, halo_result
        )
        assert input_halo == input_halo_copy
        assert input_data == input_data_copy

        # check that the calculation returns the correct values
        for (
            name,
            size,
            dtype,
            unit,
            _,
        ) in property_calculator.exclusive_sphere_properties:
            full_name = f"ExclusiveSphere/50kpc/{name}"
            assert full_name in halo_result
            result = halo_result[full_name][0]
            assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
            assert result.dtype == dtype
            assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone version of the program: just run the unit test.

    Note that this can also be achieved by running "pytest *.py" in the folder.
    """
    print("Running test_exclusive_sphere_properties()...")
    test_exclusive_sphere_properties()
    print("Test passed.")

    print("Name & Size & Unit & Type & Description \\\\")
    for (
        name,
        size,
        dtype,
        unit,
        description,
    ) in ExclusiveSphereProperties.exclusive_sphere_properties:
        unit_str = unit.__str__()
        unit_str = unit_str.replace("1.98841586e+30 kg", "M$_\\odot{}$")
        print(
            f"\\verb+{name}+ & {size} & {unit_str} & {dtype.__name__} & {description} \\\\"
        )
