
import pytest
import unittest
import numpy as np

from compute_3d.cons import model_constructor, Datfile

# @pytest.mark.usefixtures("db_class")
class Test_Modelcons(unittest.TestCase):
    def test_model_cons(self):
        pass

    def test_datfile(self):
        paths = ["Test_data/smallestrange50mm010mms.dat", "Test_data/smallestrange50mm010mmsx2.dat",
                 "Test_data/smallestrange50mm010mmsx3.dat"]

        for p in paths:
            dfile = Datfile(data=p, format="x y", z=10, start_row=1)

            self.assertEqual(dfile.datmat.shape, (48924, 3))
            self.assertEqual(np.average(dfile.z), 10)
