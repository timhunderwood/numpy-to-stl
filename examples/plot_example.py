import numpy
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import numpy_to_stl


def get_example_array():
    return numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

#
# def get_example_array():
#     return numpy.ones((100, 100))


if __name__ == "__main__":
    array = get_example_array()
    mesh = numpy_to_stl.create_surface_mesh_from_array(array, base_height=2.0)
    #mesh.save("small_example_100_100.stl")
    numpy_to_stl.plot_mesh(mesh)
