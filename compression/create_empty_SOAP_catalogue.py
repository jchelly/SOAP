#!/usr/bin/env

"""
create_empty_SOAP_catalogue.py

Create a placeholder SOAP catalogue with only empty datasets. Useful for
snapshots that do not contain any halos, since SOAP will fail to run for those.
By creating a structurally complete but empty catalogue, tools that blindly
run on all snapshots should still work, even if the SOAP catalogue does not
technically exist.

Usage:
  python3 create_empty_SOAP_catalogue.py REFERENCE SNAPSHOT OUTPUT
where REFERENCE is another SOAP catalogue for the same simulation. This is
used to figure out which datasets and meta-data should be added to the empty
SOAP catalogue. SNAPSHOT is the snapshot for which we want to create the empty
catalogue (the snapshot for which running SOAP itself fails because there are
no halos in it). OUTPUT is the name of the output file that will be created.
Note that SNAPSHOT is required to add the correct SWIFT meta-data to the SOAP
catalogue. Providing the wrong snapshot will work, but this might upset scripts
that parse the SWIFT meta-data in the empty SOAP catalogue.
"""

import h5py
import argparse
import os


def get_snapshot_index(snapshot_name):
    """
    Turn a snapshot name into a snapshot index number.

    e.g.
      flamingo_0033.hdf5 --> 33

    Should only be used on virtual files or single file snapshots, not on files
    that are part of a multi-file snapshot (e.g. flamingo_0033.25.hdf5).
    """
    name, _ = os.path.splitext(snapshot_name)
    return int(name[-4:])


class H5copier:
    """
    Functor (class that acts as a function) used to copy over groups and
    datasets from one HDF5 file to another.
    """

    def __init__(self, ifile, snapfile, ofile, snapnum):
        """
        Constructor.

        Requires the input SOAP catalogue from which we copy, the snapshot file
        from which we want to copy SWIFT meta-data, and the output file to
        which we want to copy data. Also needs the snapshot index number to
        update some of the SOAP meta-data.
        """
        self.ifile = ifile
        self.snapfile = snapfile
        self.ofile = ofile
        self.snapnum = snapnum

    def __call__(self, name, h5obj):
        """
        Functor function, i.e. what gets called when you use () on an object
        of this class. Conforms to the h5py.Group.visititems() function
        signature.

        Parameters:
         - name: Full path to a group/dataset in the HDF5 file
                  e.g. SO/200_crit/TotalMass
         - h5obj: HDF5 file object pointed at by this path
                   e.g. SO/200_crit/TotalMass --> h5py.Dataset
                        SO/200_crit --> h5py.Group
        """

        # figure out if we are dealing with a dataset or a group
        type = h5obj.__class__
        if isinstance(h5obj, h5py.Group):
            type = "group"
        elif isinstance(h5obj, h5py.Dataset):
            type = "dataset"
        else:
            raise RuntimeError(f"Unknown HDF5 object type: {name}")

        if type == "group":
            # create the group in the output file
            self.ofile.create_group(name)
            # take care of attributes:
            #  - SWIFT/Header or SWIFT/Cosmology attributes are read directly
            #    from the snapshot file
            #  - Parameters are copied, but some snapshot file specific info
            #    is updated using the correct snapshot index number for this
            #    snapshot
            #  - For all other groups we simply copy all attributes
            if name in ["SWIFT/Header", "SWIFT/Cosmology"]:
                swift_group = name.split("/")[-1]
                for attr in self.snapfile[swift_group].attrs:
                    self.ofile[name].attrs[attr] = self.snapfile[swift_group].attrs[
                        attr
                    ]
            elif name == "Parameters":
                attrs = dict(self.ifile[name].attrs)
                old_snapnum = attrs["snapshot_nr"]
                attrs["snapshot_nr"] = self.snapnum
                attrs["swift_filename"] = attrs["swift_filename"].replace(
                    f"{old_snapnum:04d}", f"{self.snapnum:04d}"
                )
                attrs["vr_basename"] = attrs["vr_basename"].replace(
                    f"{old_snapnum:04d}", f"{self.snapnum:04d}"
                )
                for attr in attrs:
                    self.ofile[name].attrs[attr] = attrs[attr]
            else:
                for attr in self.ifile[name].attrs:
                    self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]
        elif type == "dataset":
            # dataset: get the dtype and shape and create a new dataset with
            # the same name, dtype and shape, but with the length of the array
            # (shape[0]) set to 0
            dtype = h5obj.dtype
            shape = h5obj.shape
            new_shape = None
            if len(shape) == 1:
                new_shape = (0,)
            else:
                new_shape = (0, *shape[1:])
            self.ofile.create_dataset(name, new_shape, dtype)
            for attr in self.ifile[name].attrs:
                self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]


if __name__ == "__main__":
    """
    Main entry point.
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "referenceSOAP", help="Existing SOAP catalogue whose structure is copied."
    )
    argparser.add_argument(
        "snapshot", help="Snapshot file for which we want to create an empty catalogue."
    )
    argparser.add_argument("outputSOAP", help="Output catalogue file name.")
    args = argparser.parse_args()

    snapnum = get_snapshot_index(args.snapshot)

    with h5py.File(args.referenceSOAP, "r") as ifile, h5py.File(
        args.snapshot, "r"
    ) as snapfile, h5py.File(args.outputSOAP, "w") as ofile:
        h5copy = H5copier(ifile, snapfile, ofile, snapnum)
        ifile.visititems(h5copy)
