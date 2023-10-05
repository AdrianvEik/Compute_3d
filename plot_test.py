from compute_3d.cons import model_constructor

# data_folder = r"data"
# #
# # General reader for .dat file to dictionary of numpy
# # requirement:
# data: dict[np.ndarray] = r_d.Fileread(data_folder+r"/smallestrange50mm010mmsx2.dat", delimiter=",", start_row=6,
#                          head=False, dtype=float)()
#
# # call data[0] for x
# # call data[1] for z
#
# # general plotter
# fig, ax = Default(data[0], data[1], linestyle="-", marker="")()

gr = model_constructor(path="model100.npy")
print(gr.model.shape)
print(gr.model)
gr.plot_model()