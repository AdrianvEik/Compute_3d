from typing import Union
import matplotlib.pyplot as plt
import AdrianPack.Fileread as r_d


import os

import numpy as np

from cons import model_constructor, Datfile

class Model:
    def __init__(self, path: Union[str, list[str]], location: list[int] = [0]):

        self.locations = location
        if isinstance(path, str):
            datlist = self.load_from_directory(path)
            self.model = model_constructor(datlist)

        elif isinstance(path, list):
            datlist = []
            for i in path:
                datlist.append(Datfile(i, format="x z", **{"y": location[i]}))
            self.model = model_constructor(datlist)

    def load_from_directory(self, path: str, search_for: str = ".dat"):
        # Locate all .dat, .csv or .txt files in the directory
        datlist = []
        for i in os.listdir(path):
            if i.endswith(search_for):
                if len(self.locations) > 0:
                    # If the location is specified, add it to the datfile
                    datlist.append(Datfile(path + "/" + i, format="x z", y=self.locations[i])) # Will throw an error
                else:
                    # If the location is not specified, it must be in the datfile or throw an error
                    datlist.append(Datfile(path + "/" + i, format="x y z"))

        return datlist

    def load_from_file(self, path: str, **kwargs):
        # Load a single file and separate it into a list of datfiles

        datlist = []
        ddict = r_d.Fileread(path, delimiter=kwargs.get("delimiter", ","), start_row=kwargs.get("start_row", 5),
                     head=True, dtype=float)()

        # Assumed order: x, y, z
        dmat = np.array(list(ddict.values()))

        # Turn every axis into a set
        lenset = [set(dmat[0]), set(dmat[1]), set(dmat[2])]

        # The shortest set is the repeating axis
        shortest = lenset.index(min([len(i) for i in lenset]))

        # The slices are the size of the column divided by the size of the shortest set
        slices = int(len(dmat[0]) / len(lenset[shortest]))

        # Cut the matrix into slices and add them to the datlist
        for i in range(slices):
            datlist.append(Datfile({"x": dmat[0][i::slices],
                                    "y": dmat[1][i::slices],
                                    "z": dmat[2][i::slices]}),
                                    format="x y z")

        return datlist