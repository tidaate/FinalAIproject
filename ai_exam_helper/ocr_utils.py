import cv2
import pytesseract
import re
import os

# Если ты на Windows — укажи путь к tesseract.exe:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_score_from_image(image_path):
    # Чтение изображения
    image = cv2.imread(image_path)

    # Перевод в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Пороговая обработка — выделение текста
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Извлечение текста с изображения
    text = pytesseract.image_to_string(thresh, lang='eng')
    print("OCR Text:", text)

    # Поиск числа (например, 85.5)
    match = re.search(r'\d+(\.\d+)?', text)
    if match:
        return float(match.group())
    return None
