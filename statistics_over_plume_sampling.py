import os
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
from pygame.locals import *
from homemade_maths import calculate_derivative
from movement_detection import *
from ultralytics import YOLO
import supervision as sv
import geopy.distance


"""""""""""""""""""""""""""
    Fuego coordinates
"""""""""""""""""""""""""""

lat_fuego = 14.4747
lgn_fuego = -90.8806


###### Telemetry Import ######
logs_path = r"D:\Volcanology Videos\2019-04-02_3_LeiaLastFlightOfTheTrip\Flight Logs\1.BIN-3047183.mat"
logs_data = loadmat(logs_path)

"""""""""""""""""""""""""""""""""""""""""""""
        Processing of flight data
"""""""""""""""""""""""""""""""""""""""""""""

###### Extract GPS Data, IMU_Data, ARH Data ######
GPS = logs_data['GPS']
GPS_Time = GPS[:, 1]
GPS_Lat = GPS[:, 7]
GPS_Lng = GPS[:, 8]
GPS_Alt = GPS[:, 9]
GPS_Spd = GPS[:, 10]

IMU = logs_data['IMU']
IMU_Time = IMU[:, 1]
IMU_Accx = IMU[:, 5]

AHR = logs_data['AHR2']
AHR_Time = AHR[:, 1]
AHR_Pitch = AHR[:, 3]
AHR_Roll = AHR[:, 2]

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
max_value = np.max(IMU_Accx_display)
max_index = np.argmax(IMU_Accx_display)
Departure_Time = IMU_Time_display[max_index]
Departure_Time = float(int(Departure_Time))
index_logs_departure = IMU_Time_display.index(Departure_Time)

time_inside_plume = 39 + 40 + 42 + 60* (17+12 + 6)

index_inside_plume = index_logs_departure  + int(time_inside_plume)


print(GPS_Alt_display[int(index_inside_plume)])

c1 = (lat_fuego, lgn_fuego)
c2 = (GPS_Lat_display[index_inside_plume], GPS_Lng_display[index_inside_plume])

print(geopy.distance.distance(c1, c2).m)










