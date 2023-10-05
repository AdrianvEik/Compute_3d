
import pytest

from compute_3d.cons import model_constructor, Datfile

@pytest.mark.usefixtures("db_class")
class Test_Modelcons:
    def test_model_cons(self):
        gr = model_constructor(path="model100.npy")
        print(gr.model.shape)
        print(gr.model)
        gr.plot_model()

    def test_datfile(self):
        pass