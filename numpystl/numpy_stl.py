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

BAR_WIDTH = 1


def scale_and_offset(base, x, y, z, scale_x, scale_y, scale_z):
    return base * numpy.array([scale_x, scale_y, scale_z]).T + numpy.array([x, y, z])


def _get_faces_for_cell(
    x, y, cell_value, left_value, right_value, front_value, back_value, base_height
):
    # for top_face use bottom face (offset by cell_value)
    LOGGER.debug(f"for {x}, {y} --> {cell_value}")
    if (cell_value > 0) or (base_height > 0):
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


def _get_base_faces(shape, base_height:float=0):
    bottom_face = scale_and_offset(
        BOTTOM_FACE, 0, 0, -base_height, BAR_WIDTH*shape[0], BAR_WIDTH*shape[1], 1
    )
    if base_height == 0:
        return [bottom_face]

    left_face = scale_and_offset(
        LEFT_FACE, 0, 0, -base_height, BAR_WIDTH*shape[0], BAR_WIDTH*shape[1], base_height
    )
    right_face = scale_and_offset(
        RIGHT_FACE, 0, 0, -base_height, BAR_WIDTH*shape[0], BAR_WIDTH*shape[1], base_height
    )
    back_face = scale_and_offset(
        BACK_FACE, 0, 0, -base_height, BAR_WIDTH*shape[0], BAR_WIDTH*shape[1], base_height
    )
    front_face = scale_and_offset(
        FRONT_FACE, 0, 0, -base_height, BAR_WIDTH*shape[0], BAR_WIDTH*shape[1], base_height
    )

    base_faces = [bottom_face, left_face, right_face, back_face, front_face]
    return base_faces


def create_surface_stl_array(array: numpy.ndarray, base_height: float = 0) -> numpy.ndarray:
    shape = array.shape
    padded_array = numpy.zeros((shape[0] + 2, shape[1] + 2), dtype=array.dtype)
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
                BAR_WIDTH*(x - 1),
                BAR_WIDTH*(y - 1),
                cell_value,
                left_value,
                right_value,
                front_value,
                back_value,
                base_height
            )
            all_faces += faces
    all_faces += _get_base_faces(shape, base_height=base_height)
    all_faces = numpy.array(all_faces)
    all_faces = all_faces.reshape(-1, *all_faces.shape[-2:])
    data = numpy.zeros(all_faces.shape[0], dtype=stl.mesh.Mesh.dtype)
    print(all_faces.dtype)
    data["vectors"] = all_faces
    return data


def create_surface_mesh_from_array(array: numpy.ndarray, base_height: float = 0) -> stl.mesh.Mesh:
    stl_data = create_surface_stl_array(array, base_height=base_height)
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
