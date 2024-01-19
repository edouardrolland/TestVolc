import os
from merge_videos import merge_vid
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from moviepy.editor import VideoFileClip, TextClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from distance_calculation import calculate_distance
import cv2
import numpy as np
import datetime


from homemade_maths import calculate_derivative
from movement_detection import *


def merge_images(image1, image2):

    # Redimensionner la deuxième image pour qu'elle ait la même hauteur que la première image
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    if height1 != height2:
        ratio = height1 / height2
        image2 = cv2.resize(image2, (int(width2 * ratio), height1))
    # Fusionner les deux images horizontalement
    merged_image = cv2.hconcat([image1, image2])

    return merged_image




"""""""""""""""""""""""""""
    Fuego coordinates
"""""""""""""""""""""""""""

lat_fuego = 14.4747
lgn_fuego = -90.8806

""""""""""""""""""""""""""""
        Files Import
"""""""""""""""""""""""""""

###### Flight Video Import ######

A = False

if A:
    videos_path = r'C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\ExampleofFlight_without_training_data\2019-03-27_2_LeiaAutoLandOvershot\GoPro Front'
    nom_sortie = 'full_flight_640.mp4'

    if not os.path.exists(videos_path + '\\' + nom_sortie):
        print(f"The folder {videos_path} does not exist.")
        merge_vid(videos_path, nom_sortie)


videos_path = r'C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1'
nom_sortie = 'full_flight_640.mp4'

###### Telemetry Import ######
logs_path = r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\12.BIN-1610065.mat"
logs_data = loadmat(logs_path)

"""""""""""""""""""""""""""""""""""""""""""""
        Processing of flight data
"""""""""""""""""""""""""""""""""""""""""""""

###### Extract GPS Data, IMU_Data, ARH Data ######
GPS = logs_data['GPS']
GPS_Time = GPS[:, 1]
print(GPS_Time.shape)
GPS_Lat = GPS[:, 7]
GPS_Lng = GPS[:, 8]
GPS_Alt = GPS[:, 9]
GPS_Spd = GPS[:, 10]

IMU = logs_data['IMU']
IMU_Time = IMU[:, 1]
IMU_Accx = IMU[:, 5]

AHR = logs_data['ATT']
AHR_Time = AHR[:, 1]
AHR_Pitch = AHR[:, 5]
AHR_Roll = AHR[:, 3]


IMU_Time, GPS_Time = IMU_Time * 1e-6, GPS_Time * 1e-6
AHR_Time = AHR_Time * 1e-6

IMU_Time_display, GPS_Time_display = [], []
GPS_Time_display = []
GPS_Lat_display = []
GPS_Lng_display = []
GPS_Alt_display = []
GPS_Spd_display = []
IMU_Accx_display = []
AHR_Time_display = []
AHR_Pitch_display = []
AHR_Roll_display = []

###### Sequential smoothing for IMU data
IMU_Base_Time = np.around(IMU_Time[0], 0)
temp_Accx = []

for k in range(len(IMU_Time)):
    if np.around(IMU_Time[k], 0) == IMU_Base_Time:
        temp_Accx.append(IMU_Accx[k])
    else:
        IMU_Accx_display.append(np.mean(temp_Accx))
        IMU_Time_display.append(IMU_Base_Time)
        IMU_Base_Time = np.around(IMU_Time[k], 0)
        temp_Accx = []

###### Sequential smoothing for AHR data
AHR_Base_Time = np.around(AHR_Time[0], 0)
temp_Pitch = []
temp_Roll = []

for k in range(len(AHR_Time)):
    if np.around(AHR_Time[k], 0) == AHR_Base_Time:
        temp_Pitch.append(AHR_Pitch[k])
        temp_Roll.append(AHR_Roll[k])
    else:
        if len(temp_Pitch) + len(temp_Roll) == 0:
            AHR_Base_Time = np.around(AHR_Time[k], 0)
        else:
            AHR_Pitch_display.append(np.mean(temp_Pitch))
            AHR_Roll_display.append(np.mean(temp_Roll))
            AHR_Time_display.append(AHR_Base_Time)
            AHR_Base_Time = np.around(AHR_Time[k], 0)
            temp_Pitch = []
            temp_Roll = []

###### Sequential smoothing for GPS data
GPS_Base_Time = np.around(GPS_Time[0], 0)
temp_Lat = []
temp_Lng = []
temp_Alt = []
temp_Spd = []

for k in range(len(GPS_Time)):
    if np.around(GPS_Time[k], 0) == GPS_Base_Time:
        temp_Lat.append(GPS_Lat[k])
        temp_Lng.append(GPS_Lng[k])
        temp_Alt.append(GPS_Alt[k])
        temp_Spd.append(GPS_Spd[k])
    else:
        if len(temp_Lat + temp_Lng + temp_Alt + temp_Spd) == 0:
            GPS_Base_Time = np.around(GPS_Time[k], 0)
        else:
            GPS_Time_display.append(GPS_Base_Time)
            GPS_Base_Time = np.around(GPS_Time[k], 0)
            GPS_Lat_display.append(np.mean(temp_Lat))
            GPS_Lng_display.append(np.mean(temp_Lng))
            GPS_Alt_display.append(np.mean(temp_Alt))
            GPS_Spd_display.append(np.mean(temp_Spd))

        temp_Lat, temp_Lng, temp_Alt, temp_Spd = [], [], [], []

###### Take off time determinatation, logs synchronizing

# Using the accelerometer, we aim to determine the moment in the logs when Leila is catapulted
max_value = np.max(IMU_Accx)
max_index = np.argmax(IMU_Accx)
Departure_Time = IMU_Time[max_index]
Departure_Time = float(int(Departure_Time))
index_logs_departure = IMU_Time_display.index(Departure_Time)

# Machine Vision Method to detect, the timecode in the video where the drone starts to move
cap = cv2.VideoCapture(videos_path + '\\' + nom_sortie)

if (cap.isOpened()== False):
  print("Error opening video stream or file")
ret, frame = cap.read()
prev_image = frame
compteur = 0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  compteur += 1
  ret, frame = cap.read()
  if compteur%20 == 0:
    print('Looking for the drone departure in the video, frame n°' + str(compteur))
  if ret == True:
    # Display the resulting frame
    #cv2.imshow('Frame',frame)

    if detect_movement(frame, prev_image) == True:
        print("Motion detected")
        take_off_video = cap.get(cv2.CAP_PROP_POS_MSEC)
        break
        # When everything done, release the video capture object
        cap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
    else:
        prev_image = frame
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break
# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


take_off_video = round(take_off_video*1e-3)
print(take_off_video)


###### Video display

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\Test_Flight_1_output.avi")

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
previous_time = -1
AH_init_done = False


# Créer un objet de sortie vidéo
output_file = r'C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\output_done_1.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 25
frame_size = (1904, 640)
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)



while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Describe the type of font to be used
        font = cv2.FONT_HERSHEY_SIMPLEX
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time = current_time * 1e-3

        if int(current_time) != int(previous_time):
            previous_time = int(current_time)

        if current_time >= take_off_video:

            index_data = previous_time - int(take_off_video) + index_logs_departure
            flight_image = cv2.imread(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\PNG\trajectoire_" + str(index_data) + ".png")
            frame_m = merge_images(frame,flight_image)

            cv2.imshow('Frame', frame_m)
            video_writer.write(frame_m)

        else:

            start_image = cv2.imread(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\PNG\trajectoire_31.png")
            frame_m = merge_images(frame,start_image)

            cv2.imshow('Frame', frame_m)
            video_writer.write(frame_m)


        # Press 'Q' on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop if no more frames
    else:
        break

video_writer.release()
# When everything is done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()