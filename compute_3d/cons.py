## File containing constructor class for converting multiple .dat files to
## a 3D view.
from typing import Tuple, Union

import AdrianPack.Fileread as r_d

from AdrianPack.Extra import compress_ind
import matplotlib.pyplot as plt
import numpy as np


class Datfile:
    """
    Convert a .dat file to a 3 dimension nx3 numpy array.

    :param data: Path to the .dat file or dictionary of numpy arrays
                 {"x": np.ndarray, "y": np.ndarray, "z": np.ndarray}
    :param format: Format of the .dat file, e.g. "x y z"

    :param kwargs: Keyword arguments for the missing axis. e.g. x = 10 or y = 20 and for the start row in the file.
    :param kwargs: start_row: int, default = 0
    """

    def __init__(self, data: Union[str, dict], format: str, **kwargs):

        # Req, contains the required axis
        req = ["x", "y", "z"]
        fmt = [*format.replace(" ", "")]

        # Check which axis is missing, and look it up in the kwargs
        missing = list(set(req) - set(fmt))

        if isinstance(data, str):
            # Read the pth.dat file and convert it to a numpy array
            data = r_d.Fileread(data, delimiter=",", start_row=kwargs.get("start_row", 0),
                                head=True, dtype=float)()

        missing_ax = []
        # use the header to find the missing axis value
        if list(data.keys())[0] in ["x", "y", "z"]:
            missing_ax.append(list(data.keys())[1])

        # If keyerror means the axis is missing, throw an error.
        if len(missing_ax) != len(missing):
            try:
                missing_ax = [kwargs[i] for i in missing[len(missing_ax):]]
            except KeyError:
                raise Exception("Missing axis input for axis %s" % missing[-1])

        # Create a nx3 array with the missing axis at the position of the missing axis
        # e.g. if x is missing, the array will be [missing_ax, data[0], data[1]]
        # e.g. if y is missing, the array will be [data[0], missing_ax, data[1]]
        # e.g. if z is missing, the array will be [data[0], data[1], missing_ax]

        self.datmat = np.array([np.full(list(data.values())[0].size, missing_ax) if i == missing[0] else
                                list(data.values())[fmt.index(i)] for i in req]).T # Convert to nm

        # Grad for df(x)/dx, df(y)/dy, df(z)/dz
        self.grad = np.gradient(self.datmat)[0]
        self.slope_avg = [np.average(np.abs(i)) for i in self.grad.T]

    def __repr__(self):
        return "Measurement file with shape %s" % str(self.datmat.shape)

    @property
    def x(self):
        return self.datmat[:, 0]

    @property
    def y(self):
        return self.datmat[:, 1]

    @property
    def z(self):
        return self.datmat[:, 2]

    def slope_info(self) -> list[bool]:
        """
        :Return: a list of booleans, where True means the slope is approximately 0, and False means the slope is not approximately 0.
        :rtype: list[bool]
        """
        # Run a check to see if a slope is approximately 0 or not (margin of 1e-8)
        return [true if i < 1e-8 else false for i, true, false in zip(self.slope_avg, [True, True, True], [False, False, False])]

# use this to create a 3D meshgrid from multiple .dat files
class model_constructor:
    """
    Create a 3D meshgrid from multiple .dat files

    :param files: List of .dat files
    :param format: Format of the .dat file, e.g. "x y z"
    :param kwargs: Keyword arguments for the missing axis. e.g. x = 10 or y = 20


    """
    def __init__(self, datfiles: list[Datfile] = None, path: str = "", norm_data: Datfile = None):
        if isinstance(datfiles, list):
            self.datlist = datfiles
            self.norm_data = norm_data

            if norm_data is None:
                self.norm_data = 0

            slopes = [i.slope_info() for i in self.datlist]
            axes = np.where(slopes)[1]  # x = 0, y = 1, z = 2
            # Defining the data sets that belong to the horizontal and vertical axis
            # e.g. if the slope of x is 0 the data set belongs to the horizontal axis
            # e.g. if the slope of y is 0 the data set belongs to the horizontal axis

            self.axes = []

            for i in range(len(slopes)):
                if slopes[i][0] == True:
                    self.axes.append(self.datlist[i])

            # sort the horizontal and vertical axis by y and x coordinates respectively
            self.axes.sort(key=lambda x: x.x[0])

            self.model = np.zeros((self.axes[0].x.size, len(self.axes)))

        else:
            self.load_model(path)



    def __repr__(self):
        return "3D model with shape %s" % str(self.model.shape)

    def build_model(self, res: int = 10):
        # Res is the resolution of the model in pixels
        # Create a 3d numpy array containg z values for x, y coordinates
        # Indices of the array correspond to the x, y coordinates
        # So mat[x, y] = z
        zmat_data = np.zeros((self.axes[0].x.size, len(self.axes)))
        for i in range(len(self.axes)):
            zmat_data[:, i] = self.axes[i].z - self.norm_data.z

        # Create a 3d numpy array containg z values for x, y coordinates
        # Filled with extra data points to increase resolution calculated by pixel_in_square
        zmat_model = np.zeros((self.axes[0].x.size, res))

        # Now the old data is copied to the new array, with their position in this
        # matrix determined by the spacing between them casted onto the new coordinates (datalenght, res)
        for i in range(len(self.axes)):
            zmat_model[:, int(i * res / len(self.axes))] = zmat_data[:, i] / 1000 # Convert to um

        # Now the new array is interpolated to fill in the gaps using pixel_in_square
        points = np.linspace(0, self.axes[-1].x[0] - 1, res)

        for i in range(res):
            print("Line %s of %s" % (i, res))
            for j in range(zmat_model.shape[0]):
                if zmat_model[j, i] == 0:
                    zmat_model[j, i] = self.pixel_in_square([points[i], j])

        self.model = zmat_model

    def plot_model(self):
        """
        Plot a simple 3D plot of the model
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(self.model.shape[1]), range(self.model.shape[0]))
        ax.plot_surface(X, Y, self.model, rstride=100, cstride=100)
        plt.show()

    def save_model(self, filename: str):
        """
        Save the model to a .npy file

        :param filename: Name of the file to save to
        :type filename: str
        """
        np.save(filename, self.model)

    def load_model(self, filename: str):
        """
        Load a model from a .npy file

        :param filename: Name of the file to load from
        :type filename: str
        """
        self.model = np.load(filename)

    def find_square(self, pixel: Tuple[int]):
        for i in range(len(self.axes)):
            if pixel[0] < self.axes[i].x[0]:
                for j in range(self.axes[i - 1].y.size):
                    if pixel[1] < self.axes[i - 1].y[j]:
                        return [i - 1, j - 1]

        Exception("Pixel not in model")


    def pixel_in_square(self, pixel: Tuple[int]):
        """
        pixel = [ b, c]
        """

        # Determine if pixel in left or right of the line
        square = self.find_square(pixel) # ints
        square_a = [self.axes[square[0]].x[square[1]], self.axes[square[0]].y[square[1]], self.axes[square[0]].z[square[0]]] # floats
        square_b = [self.axes[square[0]].x[square[1] + 1], self.axes[square[0]].y[square[1] + 1], self.axes[square[0]].z[square[0] + 1]] # floats
        square_c = [self.axes[square[0] + 1].x[square[1]], self.axes[square[0] + 1].y[square[1]], self.axes[square[0] + 1].z[square[1]]] # floats
        square_d = [self.axes[square[0] + 1].x[square[1] + 1], self.axes[square[0] + 1].y[square[1] + 1], self.axes[square[0] + 1].z[square[1] + 1]] # floats

        conversion_to_um = self.axes[0].y.size / (self.axes[0].y[-1] * 1000)


        # rc = c1 - a1 / c0 / a0, cy - ay / cx - ax
        rc = (square_c[1] - square_a[1]) / (square_c[0] - square_a[0])

        # Left of the line
        if pixel[1] < rc * pixel[0] - rc * square_a[0] - square_a[1]:
            # z = ax + b, a =  dz/dx, b = z - a * x i.e. Az - a * Ax
            # In terms of data points
            # z = (Bz - Bx)/(Az - Ax) * x + (Bz - Bx)/(Az - Ax) * Ax + Az
            # z = ax + b + cy
            # Cz = aCx + b + c* Cy (Cx,Cy, Cz being coordinates)
            # c =( Cz - aCx - b) /  Cy
            # c = (Bz - Bx)/(Az - Ax) * (Cx - Ax) + (Bz - Bx)/(Az - Ax) * Ax + Az - aCx - b) /  Cy
            # Pz = a * Px + b + c * Py < the value interpolated from the data points
            # Pz = a * Px + b + (Bz - Bx)/(Az - Ax) * (Cx - Ax) + (Bz - Bx)/(Az - Ax) * Ax + Az - aCx - b) /  Cy * Py

            # z = ax + by + c
            b = (square_b[2] - square_a[2]) / (square_b[1] - square_a[1])
            c = square_a[2] - b * square_a[1]
            a = (square_c[2] - b * square_c[1] - c) / square_c[0]

            Pz = -(a * pixel[0] - b * pixel[1]) - c
        else:
            # The pixel is on the right of the line
            a = (square_d[2] - square_b[2]) / (square_d[0] - square_b[0])
            c = square_b[2] - a * square_b[0]
            b = (square_c[2] - a * square_c[0] - c) / square_c[1]

            Pz = -(a * pixel[0] + b * pixel[1]) - c

        if np.abs(Pz) > 10:
            print("a: %s, b: %s, c: %s" % (a, b, c))
            print("A: %s, B: %s, C: %s, D: %s" % (square_a, square_b, square_c, square_d))
            print(Pz, pixel[0], pixel[1] * conversion_to_um)
        return Pz

    @staticmethod
    def gen_eq(x, refxy, a, b, c):
        """
        :param x: x-coordinate
        :param refxy: reference point (x, y)
        :param a: slope of the line
        :param b: amount of x ref point shifts
        :param c: amount of y ref point shifts
        :return: y coordinate
        """
        # To find triangle on left of the line y < a(x - rx) + ry
        # To find triangle on right of the line y > a(x - rx) + ry
        return a * (x - b * refxy[0]) + c * refxy[1]  # y = ax + b

if __name__ == "__main__":
    file1 = Datfile("data/smallestrange50mm010mms.dat", "y z", x=0)
    file2 = Datfile("data/smallestrange50mm010mmsx2.dat", "y z", x=10)
    file3 = Datfile("data/smallestrange50mm010mmsx3.dat", "y z", x=20)
    file4 = Datfile("data/smallestrange50mm010mmsx4.dat", "y z", x=30)
    file5 = Datfile("data/smallestrange50mm010mmsx5.dat", "y z", x=40)
    file6 = Datfile("data/smallestrange50mm010mmsx6.dat", "y z", x=50)
    file7 = Datfile("data/smallestrange50mm010mmsx7.dat", "y z", x=60)

    norm_file = Datfile("data/nullmeting.dat", "y z", x=0)

    # file1 = Datfile("data/nullmeting.dat", "x z", y=0)
    # file2 = Datfile("data/largelestrange50mm010mmsx1.dat", "x z", y=200)
    # file3 = Datfile("data/largelestrange50mm010mmsx2.dat", "x z", y=400)

    gr = model_constructor([file1, file2, file3, file4, file5, file6, file7], norm_data=norm_file)
    gr.build_model(res=10)
    gr.plot_model()
    gr.save_model("data/model10.npy")

    # gr = model_constructor(path="model.npy")
    # gr.plot_model()