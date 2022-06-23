#!/bin/env python

import numpy as np
import unyt

from halo_properties import HaloProperty
from dataset_names import mass_dataset
from half_mass_radius import get_half_mass_radius


class ProjectedApertureProperties(HaloProperty):
    """
    Calculate projected aperture properties.

    These contain all particles bound to a halo. For projections along the three
    principal coordinate axes, all particles within a given fixed aperture
    radius are used. The depth of the projection is always the full extent of
    the halo along the projection axis.
    """

    # Particle properties that are used
    particle_properties = {
        "PartType0": [
            "Coordinates",
            "GroupNr_bound",
            "Masses",
            "StarFormationRates",
            "Velocities",
        ],
        "PartType1": ["Coordinates", "GroupNr_bound", "Masses", "Velocities"],
        "PartType4": [
            "Coordinates",
            "GroupNr_bound",
            "InitialMasses",
            "Luminosities",
            "Masses",
            "Velocities",
        ],
        "PartType5": [
            "Coordinates",
            "DynamicalMasses",
            "GroupNr_bound",
            "SubgridMasses",
            "Velocities",
        ],
    }

    def __init__(self, cellgrid, physical_radius_kpc):
        super().__init__(cellgrid)

        # No density criterion
        self.mean_density_multiple = None
        self.critical_density_multiple = None

        # Minimum physical radius to read in (pMpc)
        self.physical_radius_mpc = 0.001 * physical_radius_kpc

        self.name = f"projected_aperture_{physical_radius_kpc:.0f}kpc"

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
        radius_projx = []
        radius_projy = []
        radius_projz = []
        velocity = []
        types = []
        for ptype in types_present:
            grnr = data[ptype]["GroupNr_bound"]
            in_halo = grnr == index
            mass.append(data[ptype][mass_dataset(ptype)][in_halo])
            pos = data[ptype]["Coordinates"][in_halo, :] - centre[None, :]
            position.append(pos)
            rprojx = np.sqrt(pos[:, 1] ** 2 + pos[:, 2] ** 2)
            radius_projx.append(rprojx)
            rprojy = np.sqrt(pos[:, 0] ** 2 + pos[:, 2] ** 2)
            radius_projy.append(rprojy)
            rprojz = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            radius_projz.append(rprojz)
            velocity.append(data[ptype]["Velocities"][in_halo, :])
            typearr = np.zeros(rprojx.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)

        mass = unyt.array.uconcatenate(mass)
        position = unyt.array.uconcatenate(position)
        radius_projx = unyt.array.uconcatenate(radius_projx)
        radius_projy = unyt.array.uconcatenate(radius_projy)
        radius_projz = unyt.array.uconcatenate(radius_projz)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)

        mask_projx = radius_projx <= self.physical_radius_mpc * unyt.Mpc
        mask_projy = radius_projy <= self.physical_radius_mpc * unyt.Mpc
        mask_projz = radius_projz <= self.physical_radius_mpc * unyt.Mpc

        for projname, projmask, projr in zip(
            ["projx", "projy", "projz"],
            [mask_projx, mask_projy, mask_projz],
            [radius_projx, radius_projy, radius_projz],
        ):
            proj_mass = mass[projmask]
            proj_position = position[projmask]
            proj_velocity = velocity[projmask]
            proj_radius = projr[projmask]
            proj_type = types[projmask]

            gas_mask_ap = projmask[types == "PartType0"]
            star_mask_ap = projmask[types == "PartType4"]
            bh_mask_ap = projmask[types == "PartType5"]

            proj_mass_gas = proj_mass[proj_type == "PartType0"]
            proj_mass_dm = proj_mass[proj_type == "PartType1"]
            proj_mass_star = proj_mass[proj_type == "PartType4"]

            proj_Mtot = proj_mass.sum()
            proj_Mgas = proj_mass_gas.sum()
            proj_Mdm = proj_mass_dm.sum()
            proj_Mstar = proj_mass_star.sum()
            if np.any(star_mask_ap):
                star_mask_all = data["PartType4"]["GroupNr_bound"] == index
                proj_Mstar_init = data["PartType4"]["InitialMasses"][star_mask_all][
                    star_mask_ap
                ].sum()
                proj_lum = data["PartType4"]["Luminosities"][star_mask_all][
                    star_mask_ap
                ].sum(axis=0)
            else:
                proj_Mstar_init = unyt.unyt_array(proj_Mstar)
                proj_lum = unyt.unyt_array(
                    [0.0] * 9,
                    dtype=np.float32,
                    units="dimensionless",
                    registry=mass.units.registry,
                )
            proj_Mbh = proj_mass[proj_type == "PartType5"].sum()
            if np.any(bh_mask_ap):
                bh_mask_all = data["PartType5"]["GroupNr_bound"] == index
                proj_Mbh_subgrid = data["PartType5"]["SubgridMasses"][bh_mask_all][
                    bh_mask_ap
                ].sum()
            else:
                proj_Mbh_subgrid = unyt.unyt_array(proj_Mbh)

            proj_com = unyt.unyt_array(
                [0.0] * 3, dtype=np.float32, units="Mpc", registry=mass.units.registry
            )
            proj_vcom = unyt.unyt_array(
                [0.0] * 3, dtype=np.float32, units="km/s", registry=mass.units.registry
            )
            if proj_Mtot > 0.0 * proj_Mtot.units:
                proj_com[:] = (proj_mass[:, None] * proj_position).sum(
                    axis=0
                ) / proj_Mtot
                proj_com[:] += centre
                # perform the mass division before the multiplication to avoid
                # numerical overflow
                proj_vcom[:] = ((proj_mass[:, None] / proj_Mtot) * proj_velocity).sum(
                    axis=0
                )

            proj_SFR = 0.0
            if np.any(gas_mask_ap):
                gas_mask_all = data["PartType0"]["GroupNr_bound"] == index
                proj_SFR = data["PartType0"]["StarFormationRates"][gas_mask_all][
                    gas_mask_ap
                ]
                # Negative SFR are not SFR at all!
                proj_SFR = proj_SFR[proj_SFR > 0.0].sum()
            proj_SFR = unyt.unyt_array(
                proj_SFR,
                dtype=np.float32,
                units="Msun/yr",
                registry=mass.units.registry,
            )

            # sort according to radius
            halfmass = {}
            for name, r, m, M in zip(
                ["tot", "gas", "dm", "star"],
                [
                    proj_radius,
                    proj_radius[proj_type == "PartType0"],
                    proj_radius[proj_type == "PartType1"],
                    proj_radius[proj_type == "PartType4"],
                ],
                [proj_mass, proj_mass_gas, proj_mass_dm, proj_mass_star],
                [proj_Mtot, proj_Mgas, proj_Mdm, proj_Mstar],
            ):
                half_mass_radius = get_half_mass_radius(r, m, M)
                if half_mass_radius >= self.physical_radius_mpc * unyt.Mpc:
                    raise RuntimeError(
                        "Half mass radius larger than aperture"
                        f" ({half_mass_radius} >="
                        f" {self.physical_radius_mpc * unyt.Mpc}!"
                        " This should not happen."
                    )
                halfmass[name] = unyt.unyt_array(
                    half_mass_radius.value,
                    dtype=np.float32,
                    units=half_mass_radius.units,
                )

            prefix = (
                f"ProjectedAperture/{self.physical_radius_mpc*1000.:.0f}kpc/{projname}"
            )
            halo_result.update(
                {
                    f"{prefix}/Mtot": (proj_Mtot, "Total mass"),
                    f"{prefix}/Mgas": (proj_Mgas, "Total gas mass"),
                    f"{prefix}/Mdm": (proj_Mdm, "Total DM mass"),
                    f"{prefix}/Mstar": (proj_Mstar, "Total stellar mass"),
                    f"{prefix}/Mstar_init": (
                        proj_Mstar_init,
                        "Total initial stellar mass",
                    ),
                    f"{prefix}/Mbh": (proj_Mbh, "Total BH dynamical mass"),
                    f"{prefix}/Mbh_subgrid": (
                        proj_Mbh_subgrid,
                        "Total BH subgrid mass",
                    ),
                    f"{prefix}/com": (proj_com, "Centre of mass"),
                    f"{prefix}/vcom": (proj_vcom, "Centre of mass velocity"),
                    f"{prefix}/SFR": (proj_SFR, "Total SFR"),
                    f"{prefix}/Luminosity": (proj_lum, "Total luminosity"),
                    f"{prefix}/HalfMassRadiusTot": (
                        halfmass["tot"],
                        "Total half mass radius",
                    ),
                    f"{prefix}/HalfMassRadiusGas": (
                        halfmass["gas"],
                        "Total gas half mass radius",
                    ),
                    f"{prefix}/HalfMassRadiusDM": (
                        halfmass["dm"],
                        "Total DM half mass radius",
                    ),
                    f"{prefix}/HalfMassRadiusStar": (
                        halfmass["star"],
                        "Total stellar half mass radius",
                    ),
                }
            )

        return


class DummyProjectedApertureProperties(ProjectedApertureProperties):
    """
    Minimal version of ProjectedApertureProperties, only used for testing.
    """

    def __init__(self):
        self.physical_radius_mpc = 0.03


def test_projected_aperture_properties():
    """
    Unit test for the projected aperture calculation.

    Generates 100 random halos and passes them on to
    ProjectedApertureProperties::calculate().
    Tests that all expected return values are computed and have the right size,
    dtype and units.
    """

    from dummy_halo_generator import DummyHaloGenerator

    dummy_halos = DummyHaloGenerator(127)

    property_calculator = DummyProjectedApertureProperties()

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

        for name, size, dtype, unit in [
            ("ProjectedAperture/30kpc/projx/Mtot", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/Mgas", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/Mdm", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/Mstar", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/Mstar_init", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/Mbh", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/Mbh_subgrid", 1, np.float32, unyt.Msun),
            ("ProjectedAperture/30kpc/projx/com", 3, np.float32, unyt.kpc),
            ("ProjectedAperture/30kpc/projx/vcom", 3, np.float32, unyt.km / unyt.s),
            ("ProjectedAperture/30kpc/projx/SFR", 1, np.float32, unyt.Msun / unyt.yr),
            (
                "ProjectedAperture/30kpc/projx/Luminosity",
                9,
                np.float32,
                unyt.dimensionless,
            ),
            (
                "ProjectedAperture/30kpc/projx/HalfMassRadiusTot",
                1,
                np.float32,
                unyt.kpc,
            ),
            (
                "ProjectedAperture/30kpc/projx/HalfMassRadiusGas",
                1,
                np.float32,
                unyt.kpc,
            ),
            ("ProjectedAperture/30kpc/projx/HalfMassRadiusDM", 1, np.float32, unyt.kpc),
            (
                "ProjectedAperture/30kpc/projx/HalfMassRadiusStar",
                1,
                np.float32,
                unyt.kpc,
            ),
        ]:
            assert name in halo_result
            result = halo_result[name][0]
            assert (len(result.shape) == 0 and size == 1) or result.shape[0] == size
            assert result.dtype == dtype
            assert result.units.same_dimensions_as(unit.units)


if __name__ == "__main__":
    """
    Standalone mode: simply run the unit test.

    Note that this can also be achieved by running
    python3 -m pytest *.py
    in the main folder.
    """
    print("Calling test_projected_aperture_properties()...")
    test_projected_aperture_properties()
    print("Test passed.")
