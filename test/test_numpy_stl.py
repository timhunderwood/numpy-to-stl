import unittest
import numpy
import numpystl.numpy_stl as sut
import time


class TestNumpyStl(unittest.TestCase):
    def test_get_faces_for_cell(self):
        test_output = sut._get_faces_for_cell(1, 1, 2, 0, 0, 3, 1,0)
        test_output = numpy.array(test_output)
        expected_output = numpy.array(
            [
                [
                    [[1.0, 1.0, 2.0], [1.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                    [[1.0, 1.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 2.0]],
                ],
                [
                    [[1.0, 2.0, 0.0], [1.0, 2.0, 2.0], [1.0, 1.0, 2.0]],
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 2.0], [1.0, 2.0, 0.0]],
                ],
                [
                    [[2.0, 1.0, 0.0], [2.0, 1.0, 2.0], [2.0, 2.0, 0.0]],
                    [[2.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 0.0]],
                ],
                [
                    [[1.0, 1.0, 3.0], [2.0, 1.0, 3.0], [2.0, 1.0, 2.0]],
                    [[1.0, 1.0, 3.0], [1.0, 1.0, 2.0], [2.0, 1.0, 2.0]],
                ],
                [
                    [[1.0, 2.0, 1.0], [2.0, 2.0, 1.0], [1.0, 2.0, 2.0]],
                    [[2.0, 2.0, 1.0], [1.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                ],
            ]
        )
        numpy.testing.assert_array_equal(expected_output, test_output)

    def test_create_surface_stl_array(self):
        input_array = numpy.identity(3)
        test_output = sut.create_surface_stl_array(input_array, base_height = 0)
        expected_output = numpy.array(
            [
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 2.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 1.0, 1.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [2.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 2.0, 0.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 1.0, 0.0], [2.0, 1.0, 1.0], [2.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 1.0], [2.0, 1.0, 1.0], [2.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [1.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 0.0], [1.0, 2.0, 1.0], [2.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 1.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 3.0, 0.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 2.0, 1.0], [2.0, 2.0, 1.0], [2.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[1.0, 2.0, 1.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 1.0], [2.0, 2.0, 0.0], [2.0, 1.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 1.0, 1.0], [2.0, 1.0, 0.0], [2.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 1.0], [3.0, 2.0, 1.0], [2.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[3.0, 2.0, 1.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 1.0], [2.0, 3.0, 1.0], [3.0, 3.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 3.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 3.0, 0.0], [2.0, 3.0, 1.0], [2.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 0.0], [2.0, 2.0, 1.0], [2.0, 3.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[3.0, 2.0, 0.0], [3.0, 2.0, 1.0], [3.0, 3.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[3.0, 3.0, 1.0], [3.0, 2.0, 1.0], [3.0, 3.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [3.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 2.0, 0.0], [2.0, 2.0, 1.0], [3.0, 2.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[2.0, 3.0, 0.0], [3.0, 3.0, 0.0], [2.0, 3.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[3.0, 3.0, 0.0], [2.0, 3.0, 1.0], [3.0, 3.0, 1.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]],
                    [0],
                ),
                (
                    [0.0, 0.0, 0.0],
                    [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 3.0, 0.0]],
                    [0],
                ),
            ],
            dtype=[
                ("normals", "<f4", (3,)),
                ("vectors", "<f4", (3, 3)),
                ("attr", "<u2", (1,)),
            ],
        )
        numpy.testing.assert_array_equal(expected_output, test_output)

    def test_create_surface_stl_mesh(self):
        input_array = numpy.identity(10)
        mesh = sut.create_surface_mesh_from_array(input_array, base_height = 2)
        sut.plot_mesh(mesh)
        mesh.save("identity_test.stl")
