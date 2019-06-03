"""Core functionality for converting between a 2 dimensional numpy array
to an STL file.
"""
import numpy
import stl.mesh

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
BAR_WIDTH = 1.0


def _create_bar(
    x: float, y: float, width_x: float, width_y: float, height: float
) -> numpy.ndarray:
    """Create a bar with x,y widths of width and height of height offset by x,y.

    :param x:
    :param y:
    :param width_x:
    :param width_y:
    :param height:
    :return: Returned array has shape (12,3,3)
    """
    return CUBE * numpy.array([width_x, width_y, height]).T + numpy.array([x, y, 0])


def _create_bars(
    xs: numpy.ndarray,
    ys: numpy.ndarray,
    heights: numpy.ndarray,
    base_height: float = 0,
    base_padding: float = 5,
):
    """Returns a flat numpy array of bars of shape n*12,3,3 where n is the length of input arrays (e.g. xs).

    A base bar (extending into the negative z) can be inserted below all bars.

    :param xs:
    :param ys:
    :param heights:
    :param base_height:
    :return:
    """
    bars = [
        _create_bar(x, y, BAR_WIDTH, BAR_WIDTH, height)
        for x, y, height in zip(xs, ys, heights)
    ]
    if base_height > 0:
        base = _create_base(xs, ys, base_height, base_padding)
        bars.append(base)
    return numpy.array(bars)


def _create_base(
    xs: numpy.ndarray, ys: numpy.ndarray, base_height: float, base_padding: float
):
    """

    :param xs:
    :param ys:
    :param base_height:
    :param base_padding:
    :return:
    """
    base_width_x = (max(xs) + 1) + base_padding
    base_width_y = (max(ys) + 1) + base_padding
    base = _create_bar(
        -base_padding / 2, -base_padding / 2, base_width_x, base_width_y, -base_height
    )
    return base

def create_stl_mesh_from_2d_array(
    array: numpy.ndarray, base_height: float = 0.0, base_padding: float = 5.0
):
    xs, ys = numpy.meshgrid(numpy.arange(array.shape[1]), numpy.arange(array.shape[0]))
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()
    heights_flat = array.flatten()
    bars = _create_bars(
        xs_flat,
        ys_flat,
        heights_flat,
        base_height=base_height,
        base_padding=base_padding,
    )
    print(bars.shape)
    bars = bars.reshape(-1, *bars.shape[-2:])
    print(bars.shape)
    if base_height > 0:
        vertices = (array.size + 1) * 12  # +1 for base bar
    else:
        vertices = array.size * 12
    data = numpy.zeros(vertices, dtype=stl.mesh.Mesh.dtype)
    data["vectors"] = bars
    return data
