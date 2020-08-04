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
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_segments = cv2.erode(segments, erosion_kernel, iterations=3)

    # Aproximando cantidad de grama contando secciones cerradas
    contours, hierarchy = cv2.findContours(eroded_segments, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bgr_eroded_segments = cv2.cvtColor(eroded_segments, cv2.COLOR_GRAY2BGR)
    segments_contours = cv2.drawContours(bgr_eroded_segments, contours, -1, (0, 255, 0), thickness=2)

    return len(contours), segments_contours


def tamaño_grama_exp02(img: np.ndarray) -> (float, float, np.ndarray):
    """Esta funcion implementa el metodo descrito paso a paso en el cuaderno `experimento02.ipynb`"""

    # Suavizado y binarización
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    gray = as_grayscale(blur)
    ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erosión y dilatación
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    eroded = cv2.erode(otsu, morph_kernel, iterations=4)
    dilated = cv2.dilate(eroded, morph_kernel, iterations=4)

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Usar contornos para encontrar rectángulos de área mínima
    rects = []

    for contour in contours:
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        box = np.int32(box)
        rects.append(box)

    # Aislar los rectangulos que mejor representen las hojas de grama
    def box_sides(box: np.ndarray) -> (np.float64, np.float64):
        x0, y0 = box[0]
        x1, y1 = box[1]
        x2, y2 = box[2]
        d1 = np.sqrt(np.square(x1 - x0) + np.square(y1 - y0))
        d2 = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
        return d1, d2

    mat = dilated
    rects_hoja = []

    for rect in rects:
        # proporcion del rectangulo que caracterize a una hoja de grama
        d1, d2 = box_sides(rect)
        tiene_proporcion_hoja = (d1 > d2 * 2) if d1 > d2 else (d2 > d1 * 2)

        if tiene_proporcion_hoja:
            # porcentaje de blanco
            negro = 0
            blanco = 0
            x, y, w, h = cv2.boundingRect(rect)
            for (i, j) in ((i, j) for i in range(y, min(y + h, mat.shape[0] - 1))
                           for j in range(x, min(x + w, mat.shape[1] - 1))):
                if cv2.pointPolygonTest(rect, (j, i), False) > 0:
                    color = mat[i, j]
                    if color == 0:
                        negro += 1
                    else:
                        blanco += 1

            porcentaje_blanco = blanco / (negro + blanco)

            if porcentaje_blanco > 0.72:
                rects_hoja.append(rect)

    bgr_mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    for rect_hoja in rects_hoja:
        cv2.drawContours(bgr_mat, [rect_hoja], 0, (0, 0, 255), 2)

    # Obtener el ancho de las hojas de grama más reconocibles dentro de la imagen
    ancho = None

    if len(rects_hoja) > 0:
        max_area = 0
        max_rect_hoja = None
        for rect_hoja in rects_hoja:
            d1, d2 = box_sides(rect_hoja)
            area = d1 * d2
            if area > max_area:
                max_area = area
                max_rect_hoja = rect_hoja

        if max_area > 0:
            d1, d2 = box_sides(max_rect_hoja)
            ancho = d1 if d1 < d2 else d2

    # Estimar tamaño del campo de grama de la imagen
    if ancho > 0:
        px = ancho
        cm = px * 0.7 / 37
        escala = 0.7 / cm
        area_img_cm2 = (img.shape[0] * img.shape[1]) * (np.square(0.7 / 37)) * escala

        # Escala del campo de golf en relacion a la imagen de entrada
        area_golf_1m2 = 1  # metros cuadrados
        escala_campo_golf_1m2 = area_golf_1m2 / (area_img_cm2 / 10_000)

        return area_img_cm2, escala_campo_golf_1m2, bgr_mat

    return 0.0, 0.0, bgr_mat


def experimento01(nombre: str, img: np.ndarray):
    count_grama, contornos_grama = contar_grama_exp01(img)
    plt_show2(img, contornos_grama, title1=f"exp01 - {nombre}", title2=f"hojas = {count_grama}", gui=True)


def experimento02(nombre: str, img: np.ndarray):
    area_img_cm2, escala_campo_golf_1m2, rects_hoja = tamaño_grama_exp02(img)
    count_grama, _ = contar_grama_exp01(img)
    count_grama_campo_golf = escala_campo_golf_1m2 * 300_000 * count_grama
    plt_show2(img, rects_hoja,
              title1=f"exp02 - {nombre} (hojas = {count_grama:.2f})",
              title2=f"area = {area_img_cm2:.2f} cm^2 \n "
                     f"escala campo_golf 300,000 m^2 = {escala_campo_golf_1m2 * 300_000:.2f}x \n "
                     f"hojas de grama en 300,000 m^2 = {count_grama_campo_golf:.2f}", gui=True)


def main():
    grama1 = read_jpg("grama1")
    grama2 = read_jpg("grama2")
    grama3 = read_jpg("grama3")

    experimento01("grama1", grama1)
    experimento01("grama2", grama2)
    experimento01("grama3", grama3)
    plt.show()

    experimento02("grama1", grama1)
    experimento02("grama2", grama2)
    experimento02("grama3", grama3)
    plt.show()


if __name__ == '__main__':
    print('PDAE - CV - Proyecto - Ernesto Menendez - 20072392')
    main()
