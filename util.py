import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

input_path = "data/input"
output_path = "data/output"


def read_img(name: str, flags=cv2.IMREAD_COLOR, ext=".jpg") -> np.ndarray:
    input_img_path = os.path.join(input_path, name + ext)
    img = cv2.imread(input_img_path, flags)
    if img is not None:
        return img
    else:
        raise Exception("No es posible cargar la imagen: " + input_img_path)


def read_png(name: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    return read_img(name, flags=flags, ext=".png")


def read_jpg(name: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    return read_img(name, flags=flags, ext=".jpg")


def write_jpg(name: str, image: np.ndarray, params=None, show=False):
    output_img_path = os.path.join(output_path, name + ".jpg")
    cv2.imwrite(output_img_path, image, params)
    if show:
        cv2.imshow(name, image)


def plt_show(image: np.ndarray, title=None, cmap=None, transform=lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2RGB), gui=False):
    if transform is None:
        def transform(i): return i
    fig = plt.figure(title)
    ax = plt.imshow(transform(image), cmap=cmap)
    if title:
        ax.axes.set_title(title)
    if gui:
        fig.show()


def plt_show2(image1: np.ndarray, image2: np.ndarray, title1=None, title2=None, cmap=None, transform=lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2RGB), gui=False):
    if transform is None:
        def transform(i): return i

    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    ax[0].imshow(transform(image1), cmap=cmap)
    ax[1].imshow(transform(image2), cmap=cmap)
    if title1:
        ax[0].set_title(title1)
    if title2:
        ax[1].set_title(title2)
    if gui:
        fig.show()


def is_grayscale(mat: np.ndarray) -> bool:
    return (len(mat.shape) == 2) or (len(mat.shape) == 3 and mat.shape[2] == 1)


def is_color(mat: np.ndarray):
    return len(mat.shape) == 3 and mat.shape[2] == 3


def assert_color(mat: np.ndarray):
    assert is_color(mat), "La matriz no es de profundidad 3 (color)"


def assert_grayscale(mat: np.ndarray):
    assert is_grayscale(mat), "La matriz no es de profundidad 1 (grises)"


def assert_same_shape(mat1: np.ndarray, mat2: np.ndarray):
    assert mat1.shape[0] == mat2.shape[0] and mat1.shape[1] == mat2.shape[1], \
        f"Las matrices no tienen la misma forma: " \
        f"({mat1.shape[0]}, {mat1.shape[1]}) != ({mat2.shape[0]}, {mat2.shape[1]})"


def assert_dtype(arr: np.ndarray, dtype):
    assert arr.dtype == dtype, f"Esperando dtype={dtype}"


def as_grayscale(mat: np.ndarray) -> np.ndarray:
    if is_color(mat):
        return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    else:
        assert_grayscale(mat)
        return mat
