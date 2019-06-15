"""Core functionality for converting between a 2 dimensional numpy array
to an STL file.
"""
import numpy
import stl.mesh
import logging

logging.basicConfig()
LOGGER = logging.getLogger(__file__)

# each sub-array defines a triangle (two triangles per face of cube)
CUBE = numpy.array(
    [
        [[0, 1, 1], [1, 0, 1], [0, 0, 1]],  # top face
        [[1, 0, 1], [0, 1, 1], [1, 1, 1]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 0]],  # front face
        [[1, 1, 1], [1, 0, 1], [1, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1]],  # left face
        [[0, 0, 0], [0, 0, 1], [1, 0, 1]],
        [[0, 1, 0], [1, 1, 0], [0, 1, 1]],  # right face
        [[1, 1, 0], [0, 1, 1], [1, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [0, 0, 1]],  # back face
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0]],  # bottom face
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
    ]
)
TOP_FACE = numpy.array(
    [[[0, 1, 1], [1, 0, 1], [0, 0, 1]], [[1, 0, 1], [0, 1, 1], [1, 1, 1]]]
)
RIGHT_FACE = numpy.array(
    [[[1, 0, 0], [1, 0, 1], [1, 1, 0]], [[1, 1, 1], [1, 0, 1], [1, 1, 0]]]
)
BACK_FACE = numpy.array(
    [[[0, 0, 0], [1, 0, 0], [1, 0, 1]], [[0, 0, 0], [0, 0, 1], [1, 0, 1]]]
)
FRONT_FACE = numpy.array(
    [[[0, 1, 0], [1, 1, 0], [0, 1, 1]], [[1, 1, 0], [0, 1, 1], [1, 1, 1]]]
)
LEFT_FACE = numpy.array(
    [[[0, 1, 0], [0, 1, 1], [0, 0, 1]], [[0, 0, 0], [0, 0, 1], [0, 1, 0]]]
)
BOTTOM_FACE = numpy.array(
    [[[0, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 0, 0], [1, 0, 0], [1, 1, 0]]]
)

NEIGHBOUR_FACES = (LEFT_FACE, RIGHT_FACE, BACK_FACE, FRONT_FACE)

BAR_WIDTH = 1.0


def scale_and_offset(base, x, y, z, scale_x, scale_y, scale_z):
    return base * numpy.array([scale_x, scale_y, scale_z]).T + numpy.array([x, y, z])


def _get_faces_for_cell(
    x, y, cell_value, left_value, right_value, front_value, back_value
):
    # for top_face use bottom face (offset by cell_value)
    LOGGER.debug(f"for {x}, {y} --> {cell_value}")
    if cell_value > 0:
        top_face = scale_and_offset(
            BOTTOM_FACE, x, y, cell_value, BAR_WIDTH, BAR_WIDTH, 1
        )
        faces = [top_face]
    else:
        faces = []
    neighbour_values = (left_value, right_value, front_value, back_value)
    LOGGER.debug(neighbour_values)
    for base, neighbour_value in zip(NEIGHBOUR_FACES, neighbour_values):
        delta = cell_value - neighbour_value
        if delta == 0:  # don't store walls of zero height
            LOGGER.debug("skipping")
            continue
        face = scale_and_offset(
            base,
            x,
            y,
            neighbour_value,
            BAR_WIDTH,
            BAR_WIDTH,
            (cell_value - neighbour_value),
        )
        faces.append(face)
    return faces


def _get_base_face(shape):
    bottom_face = scale_and_offset(BOTTOM_FACE, 0, 0, 0, shape[0], shape[1], 1)
    return bottom_face


def create_surface_stl_array(array: numpy.ndarray) -> numpy.ndarray:
    shape = array.shape
    padded_array = numpy.zeros((shape[0] + 2, shape[1] + 2))
    padded_array[1:-1, 1:-1] = array
    all_faces = []
    for x in range(1, shape[0] + 1):
        for y in range(1, shape[1] + 1):
            cell_value = padded_array[x, y]
            left_value = padded_array[x - 1, y]
            right_value = padded_array[x + 1, y]
            front_value = padded_array[x, y - 1]
            back_value = padded_array[x, y + 1]
            # account for padding by subtracting 1 when passing x y coords
            faces = _get_faces_for_cell(
                x - 1,
                y - 1,
                cell_value,
                left_value,
                right_value,
                front_value,
                back_value,
            )
            all_faces += faces
    all_faces.append(_get_base_face(shape))
    all_faces = numpy.array(all_faces)
    all_faces = all_faces.reshape(-1, *all_faces.shape[-2:])
    data = numpy.zeros(all_faces.shape[0], dtype=stl.mesh.Mesh.dtype)
    data["vectors"] = all_faces
    return data


def create_surface_mesh_from_array(array: numpy.ndarray) -> stl.mesh.Mesh:
    stl_data = create_surface_stl_array(array)
    return stl.mesh.Mesh(stl_data, remove_empty_areas=False)


def plot_mesh(mesh):
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d

    figure = plt.figure()
    axes = mpl_toolkits.mplot3d.Axes3D(figure)

    axes.add_collection3d(
        mpl_toolkits.mplot3d.art3d.Poly3DCollection(
            mesh.vectors, facecolor="red", edgecolor="black"
        )
    )

    scale = mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    plt.show()
