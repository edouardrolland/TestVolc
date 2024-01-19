import cv2
import time
import numpy as np

def detect_movement(image, prev_image):

    image_area = image.shape[0] * image.shape[1]
    threshold_detection = 0.04

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray', gray)
    #cv2.waitKey(0)
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(r"C:\Users\edoua\Desktop\1.jpg", prev_gray)
    #cv2.imwrite(r"C:\Users\edoua\Desktop\1_bis.jpg", gray)
    #cv2.imshow('Prev Gray', prev_gray)
    #cv2.waitKey(0)
    # Calculer la différence avec l'image précédente
    diff = cv2.absdiff(prev_gray, gray)
    if __name__ == "__main__":
        cv2.imshow('Difference', diff)
        cv2.imwrite(r"C:\Users\edoua\Desktop\2.jpg", diff)
        cv2.waitKey(0)
    # Threshold
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    if __name__ == "__main__":
        cv2.imshow('Threshold', threshold)
        cv2.imwrite(r"C:\Users\edoua\Desktop\3.jpg", threshold)
        cv2.waitKey(0)
    # Opening to reduce the noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    if __name__ == "__main__":
        cv2.imshow('Opening', threshold)
        cv2.imwrite(r"C:\Users\edoua\Desktop\4.jpg", threshold)
        cv2.waitKey(0)
    # detection of the contours of regions where there is a change
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res_contours = []
    for contour in contours:
            res_contours.append(cv2.contourArea(contour))

    # Display the contours
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
    if __name__ == "__main__":
        cv2.imshow('Contours', image_with_contours)
        #cv2.imwrite(r"C:\Users\edoua\Desktop\5.jpg", image_with_contours)
        cv2.waitKey(0)

    moved_surface = float(np.sum(res_contours)/image_area)

    if moved_surface > float(threshold_detection):
        return True
    else:
        return False

if __name__ == "__main__":
    image1 = cv2.imread(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\drone_volcanic_monitoring\Drone_Volcanic_Monitoring\Movement Detection\Image1.jpg")
    image2 = cv2.imread(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\drone_volcanic_monitoring\Drone_Volcanic_Monitoring\Movement Detection\Image2.jpg")
    image3 =  cv2.imread(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\drone_volcanic_monitoring\Drone_Volcanic_Monitoring\Movement Detection\Image3.jpg")
    detect_movement(image2,image1)
    detect_movement(image3,image2)




