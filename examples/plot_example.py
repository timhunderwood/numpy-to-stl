import numpy
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import numpystl


def plot_mesh(array, base_height: float = 0.0, base_padding: float = 5.0):
    mesh = numpystl.create_stl_mesh_from_2d_array(array, base_height, base_padding)
    figure = plt.figure()
    axes = mpl_toolkits.mplot3d.Axes3D(figure)
    #
    # # Load the STL files and add the vectors to the plot
    axes.add_collection3d(
        mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            mesh.vectors, facecolor="red", edgecolor="black"
        )
    )

    # Auto scale to the mesh size
    scale = mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()


def get_example_array():
    return numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


if __name__ == "__main__":
    array = get_example_array()
    plot_mesh(array, base_height=0)
