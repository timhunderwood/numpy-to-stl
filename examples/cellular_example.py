import cellular
import numpy
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import numpy_to_stl


def get_simulated_world(cells_per_day, rule, number_of_days):
    world = cellular.World(cells_per_day, rule, ones=False)
    world.simulate(number_of_days)
    world.display()
    return numpy.vstack(world.state)


def create_mesh_of_world(
    cells_per_day=100, rule=cellular.rules.rule_777, number_of_days=100
):
    array = get_simulated_world(cells_per_day, rule, number_of_days)
    return numpy_to_stl.create_surface_mesh_from_array(array, base_height=1, )


def plot_stl_world(cells_per_day=100, rule=cellular.rules.rule_777, number_of_days=200):
    world_mesh = create_mesh_of_world(cells_per_day, rule, number_of_days)

    figure = plt.figure()
    axes = mpl_toolkits.mplot3d.Axes3D(figure)
    #
    # # Load the STL files and add the vectors to the plot
    axes.add_collection3d(
        mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            world_mesh.vectors, facecolor="red", edgecolor="black"
        )
    )

    # Auto scale to the mesh size
    scale = world_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()
    world_mesh.save("small_cellular_example.stl")


if __name__ == "__main__":
    plot_stl_world()
