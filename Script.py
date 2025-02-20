
import cv2
import numpy as np
from PIL import Image as PILImage

import os.path
import sys

import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.withdraw()

OUTPUT_FILE_NAME = r"final_image.jpg"

def load_images():
    file_paths = filedialog.askopenfilenames(title="Выберите изображения", filetypes=(("JPG", "*.jpg;*.jpeg"), ("PNG", "*.png"), ("All files", "*.*")))
    # Если пользователь нажал "Отмена" (пустой список)
    if not file_paths: 
        sys.exit()  # Завершаем программу

    if len(file_paths) != 3:
        messagebox.showwarning("Ошибка", "Необходимо выбрать 3 файла")
        sys.exit()  # Завершаем программу

    return file_paths


class Image:
    def __init__(self, path):
        self.path = path # Публичный атрибут path, задается при создании объекта
        self.read_image()

    def read_image(self):

        # Открытие изображения с помощью Pillow
        image = PILImage.open(self.path).convert('RGB')

        # Преобразование изображения в формат numpy массива
        image_np = np.array(image)

        # Преобразование из RGB в BGR
        self.data = image_np[:, :, ::-1]

        #self.data = cv2.imread(self.path, cv2.IMREAD_COLOR)
        self.type = self.detect_type()

    def detect_type(self):
        """Определяет тип изображения."""
        # Проверяем, является ли изображение черно-белым
        if self.is_grayscale():
            return "contour"    
        elif self.is_highlight():
            return "highlight"
        else:
            return "color"

    def is_grayscale(self):
        """Проверяет, является ли изображение черно-белым."""
        # Проверяем, равны ли каналы B, G и R
        b, g, r = cv2.split(self.data)
        return np.allclose(b, g) and np.allclose(g, r)

    def is_highlight(self, color_ratio_threshold=0.7):
        """Проверяет, является ли изображение изображением с выделением (highlight)."""
        
        # Разделяем изображение на каналы BGR
        b, g, r = cv2.split(self.data)
        
        # Создаём маску для белых пикселей (значения всех каналов близки к 255)
        white_mask = (r == 255) & (g == 255) & (b == 255)
        
        # Вычисляем долю белых пикселей
        white_ratio = np.mean(white_mask)
        print("white ratio", white_ratio)
        # Если доля белых пикселей больше порога, то это highlight
        return white_ratio > color_ratio_threshold
    
def create_red_mask(highlight):
    highlight_hsv = cv2.cvtColor(highlight.data, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])  # Красный нижний порог
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])  # Красный верхний порог
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(highlight_hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(highlight_hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2  # Комбинированная маска
    inverse_red_mask = cv2.bitwise_not(red_mask)
    return inverse_red_mask

def get_contour_mask(image_data):
    # Добавляем чёрные пиксели из contour на color
    gray_contour = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    black_mask = cv2.inRange(gray_contour, 0, 10)  # Порог для чёрных пикселей
    
    return black_mask

def blend_with_mask(base_image, overlay_image, mask):
    """
    Простое смешивание двух изображений с использованием маски.
    
    :param base_image: numpy.ndarray, базовое изображение
    :param overlay_image: numpy.ndarray, накладываемое изображение
    :param mask: numpy.ndarray, маска (где применять смешивание)
    :return: numpy.ndarray, изображение с применённым смешиванием
    """
    blended = cv2.addWeighted(base_image[mask > 0], 0.5, overlay_image[mask > 0], 0.5, 0)
    result = base_image.copy()
    result[mask > 0] = blended
    return result

file_paths = load_images()
images = [Image(path) for path in file_paths]

for image in images:
    if image.type == "color":
        color = image
    elif image.type == "contour":
        contour = image
    elif image.type == "highlight":
        highlight = image

output_directory = os.path.dirname(file_paths[0])
output_path = os.path.join(output_directory, OUTPUT_FILE_NAME)

white_background = np.full_like(color.data, (255, 255, 255), dtype=np.uint8)

# Определяем маску красного цвета
inverse_red_mask = create_red_mask(highlight)

# Маска для прозрачности (прозрачность 50% для красных пикселей)
color_transparent = blend_with_mask(color.data, white_background, inverse_red_mask)

#Маска контуров
contour_mask = get_contour_mask(contour.data)

# Наложение контуров на изображение
final_image = blend_with_mask(color_transparent, contour.data, contour_mask)

# Преобразуем numpy массив (OpenCV) в формат, с которым работает PIL
pil_image = PILImage.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
pil_image.save(output_path)

os.startfile(output_directory)

#cv2.imwrite(output_path, final_image)
print(f"Изображение сохранено как {OUTPUT_FILE_NAME}.jpg")
