import numpy
import imageio
import matplotlib.pyplot as plt
import numpy_to_stl


def convert_raw_image(file_name):
    array = imageio.imread(file_name)
    cropped = array[210:-240, 210:-240]
    imageio.imwrite(file_name[0:-4] + "_cropped.png", cropped)


def normalise_camera_image(array):
    scale = 10089.33
    offset = 5000.0
    optical_density = (array - offset) / scale
    optical_density = numpy.clip(optical_density, a_min=0, a_max=4)
    z_scale_factor = 0.25 * 0.5 * max(array.shape) / optical_density.max()
    scaled_array = z_scale_factor * optical_density
    print(scaled_array.shape)
    return scaled_array


if __name__ == "__main__":
    numpy_to_stl.numpy_to_stl.BAR_WIDTH = 0.25
    convert_raw_image("raw_075.png")
    raw = imageio.imread("raw_075_cropped.png")
    od = normalise_camera_image(raw)
    mesh = numpy_to_stl.create_surface_mesh_from_array(od, base_height=5)
    plt.matshow(od)
    plt.show()
    mesh.save("bec_surface_with_base_scaled.stl")
