import numpy as np
import cv2
import os
from pathlib import Path
import logging


def filter_contours(contours, sensitivity=.1):
    """
    filter contours based on their area
    :param contours: Python list of contours, as return from openCV findContours
    :param sensitivity: sensitivity parameter. 0 = all contours, >1 = none
    :return: list of filtered contours
    """
    contours_sizes = np.array([cv2.contourArea(contour) for contour in contours])
    max_size = np.max(contours_sizes)
    filter_small = contours_sizes > max_size * sensitivity
    filtered_contours = []
    for i, val in enumerate(filter_small):
        if val:
            filtered_contours.append(contours[i])
    return filtered_contours


def split(source_director):
    """
    Iterate over all photos in the directory and split them
    :param source_director: traget directory
    :return: none
    """
    logging.basicConfig(filename='filtered_photos.log', level=logging.INFO, format='%(message)s')
    total_photos = 0
    logging.info(f'Processing {source_director} directory')
    for file in os.listdir(source_director):
        if not file.startswith('.'):  # ignore hidden files
            full_file_name = os.fsdecode(file)
            file_path = source_director + '/' + full_file_name
            file_name = os.path.splitext(file_path)[0].split('/')[-1]
            directory_path = f"cropped/{file_name}_output/"
            Path(directory_path).mkdir(parents=True, exist_ok=True)

            im = cv2.imread(file_path)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 245, 250, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = filter_contours(contours)
            for i, contour in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(contour)
                cropped = im[y:y + h, x:x + w]
                cv2.imshow('crop', cropped)
                cv2.imwrite(f'{directory_path}{file_name}_{i + 1}.jpg', cropped)
            logging.info(f'file {file_name} : {len(filtered_contours)} photos')
            total_photos += len(filtered_contours)
    logging.info(f'Total number of photos : {total_photos}')


if __name__ == '__main__':
    split('source')
