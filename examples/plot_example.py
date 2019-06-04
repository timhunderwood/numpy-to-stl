import numpy
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import numpystl


def get_example_array():
    return numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


if __name__ == "__main__":
    array = get_example_array()
    mesh = numpystl.create_stl_mesh_from_2d_array(array, base_height=0.0)
    numpystl.plot_mesh(mesh)
