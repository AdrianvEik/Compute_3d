
import pytest

from compute_3d.cons import model_constructor, Datfile

@pytest.mark.usefixtures("db_class")
class Test_Modelcons:
    def test_model_cons(self):
        pass

    def test_datfile(self):
        dfile = Datfile(path="Test_data/smallestrange50mm010mmsx2.dat",
                        format="x y", z=10)

        self.assertEqual(dfile.datmat.shape, (2, 1000))
