import numpy as np
import cv2
import os
from pathlib import Path
import math


def rotate_image(image, angle):
    """
    rotate the image by an angle
    :param image: original image
    :param angle: rotation angle
    :return: rotated image, with white background color
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return result


def find_rotation_angle(im):
    """
    Apply Canny edge detection and Probabilistic Hough Lines, then find the median rotation angle
    :param im: image
    :return: rotation angle in degrees
    """
    dst = cv2.Canny(im, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 80, None, 50, 10)

    angles = []
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    angles = np.array(angles)
    angles1 = angles[np.abs(angles) < 45]
    median1 = np.median(angles1)
    angles2 = angles[np.abs(angles) >= 45]
    median2 = np.median(angles2)

    if np.abs(median1) > np.abs(90 - np.abs(median2)):
        return median1
    return np.abs(90 - np.abs(median2)) * np.sign(-median2)


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


def split(source_directory):
    """
    Iterate over all photos in the directory and split them
    :param source_directory: traget directory
    :return: none
    """
    log = ''
    total_photos = 0
    log += f'Processing {source_directory} directory\n'
    for file in os.listdir(source_directory):
        if not file.startswith('.'):  # ignore hidden files
            full_file_name = os.fsdecode(file)
            file_path = source_directory + '/' + full_file_name
            file_name = os.path.splitext(file_path)[0].split('/')[-1]
            directory_path = f"{source_directory}_split/{file_name}_output/"
            Path(directory_path).mkdir(parents=True, exist_ok=True)

            im = cv2.imread(file_path)
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 245, 250, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = filter_contours(contours)
            log += f'file {file_name} : {len(filtered_contours)} photos\n'
            for i, contour in enumerate(filtered_contours):
                x, y, w, h = cv2.boundingRect(contour)
                cropped = im[y:y + h, x:x + w]
                cropped_opening = opening[y:y + h, x:x + w]
                angle = find_rotation_angle(cropped_opening)
                rotated = rotate_image(cropped, angle)
                cv2.imwrite(f'{directory_path}{file_name}_{i + 1}.jpg', rotated)
                log += f'\tsub-photo {i + 1}: rotation = {angle:.3f}\n'
            total_photos += len(filtered_contours)
    log += f'Total number of photos : {total_photos}'

    f = open('log.txt', 'w+')
    f.write(log)
    f.close()


if __name__ == '__main__':
    split('source')
