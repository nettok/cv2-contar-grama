from util import read_jpg, as_grayscale, plt_show2

import matplotlib.pyplot as plt
import numpy as np
import cv2


def contar_grama_exp01(img: np.ndarray) -> (int, np.ndarray):
    """Esta funcion implementa el metodo descrito paso a paso en el cuaderno `experimento01.ipynb`"""

    # Aumentar contraste y suavizar la imagen
    blurred = cv2.medianBlur(img, 5)
    blurred = cv2.convertScaleAbs(blurred, dst=blurred, alpha=1.5, beta=-50)
    blurred = cv2.medianBlur(blurred, 5, dst=blurred)

    # Detección de bordes
    grayscale = as_grayscale(blurred)
    edges = cv2.Canny(grayscale, 10, 200, 5, L2gradient=True)

    # Dilatación de bordes
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, dilation_kernel, iterations=1)

    # Negativo y erosion de segmentos
    segments = np.logical_not(dilated.astype(np.bool)).astype(np.uint8) * 255
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded_segments = cv2.erode(segments, erosion_kernel, iterations=1)

    # Aproximando cantidad de grama contando secciones cerradas
    contours, hierarchy = cv2.findContours(eroded_segments, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bgr_eroded_segments = cv2.cvtColor(eroded_segments, cv2.COLOR_GRAY2BGR)
    segments_contours = cv2.drawContours(bgr_eroded_segments, contours, -1, (0, 255, 0), thickness=2)

    return len(contours), segments_contours


def main():
    grama1 = read_jpg("grama1")
    grama2 = read_jpg("grama2")
    grama3 = read_jpg("grama3")

    # Experimento #01

    count_grama1, contornos_grama1 = contar_grama_exp01(grama1)
    count_grama2, contornos_grama2 = contar_grama_exp01(grama2)
    count_grama3, contornos_grama3 = contar_grama_exp01(grama3)

    plt_show2(grama1, contornos_grama1, title1="exp01 - grama1", title2=f"hojas = {count_grama1}", gui=True)
    plt_show2(grama2, contornos_grama2, title1="exp01 - grama2", title2=f"hojas = {count_grama2}", gui=True)
    plt_show2(grama3, contornos_grama3, title1="exp01 - grama3", title2=f"hojas = {count_grama3}", gui=True)

    plt.show()


if __name__ == '__main__':
    print('PDAE - CV - Proyecto - Ernesto Menendez - 20072392')
    main()
