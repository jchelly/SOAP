#!/bin/env python3

import halo_centres
import swift_cells

# Location of the input
vr_basename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/vr/catalogue_0013/vr_catalogue_0013.properties"
swift_filename = "/cosma8/data/dp004/jch/FLAMINGO/BlackHoles/200_w_lightcone/snapshots/flamingo_0013.hdf5"

# Read the halo catalogue
so_cat = halo_centres.SOCatalogue(vr_basename)

# Read SWIFT cells
cellgrid = swift_cells.SWIFTCellGrid(swift_filename)

# Decide on quantities to read
property_names = {
    "PartType1" : {"Coordinates", "Masses"},
}

# Determine which cells to read
pos_min=(0,0,0)
pos_max=(100,100,100)
mask = cellgrid.empty_mask()
cellgrid.mask_region(mask, pos_min, pos_max)

# Read the cells
data = cellgrid.read_masked_cells(property_names, mask)
