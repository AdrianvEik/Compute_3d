from typing import Union
import matplotlib.pyplot as plt

import os

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
                    datlist.append(Datfile(path + "/" + i, format="x z", y=self.locations[i]))
                else:
                    # If the location is not specified, it must be in the datfile or throw an error
                    datlist.append(Datfile(path + "/" + i, format="x z"))

        return datlist

