import os
from merge_videos import merge_vid
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from PIL import Image, ImageDraw
import geopy.distance
import folium
from tqdm import tqdm
import io
import time
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import io
from PIL import Image

from minimap import*


"""""""""""""""""""""""""""
    Fuego coordinates
"""""""""""""""""""""""""""

lat_fuego = 14.4747
lgn_fuego = -90.8806

""""""""""""""""""""""""""""
        Files Import
"""""""""""""""""""""""""""

###### Flight Video Import ######

videos_path = r'C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_2'
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
AHR_Yaw = AHR[:,7]

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
AHR_Yaw_display = []

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
temp_Yaw  = []

for k in range(len(AHR_Time)):
    if np.around(AHR_Time[k], 0) == AHR_Base_Time:
        temp_Pitch.append(AHR_Pitch[k])
        temp_Roll.append(AHR_Roll[k])
        temp_Yaw.append(AHR_Yaw[k])
    else:
        if len(temp_Pitch) + len(temp_Roll) + len(temp_Yaw) == 0:
            AHR_Base_Time = np.around(AHR_Time[k], 0)
        else:
            AHR_Pitch_display.append(np.mean(temp_Pitch))
            AHR_Roll_display.append(np.mean(temp_Roll))
            AHR_Yaw_display.append(np.mean(temp_Yaw))
            AHR_Time_display.append(AHR_Base_Time)
            AHR_Base_Time = np.around(AHR_Time[k], 0)
            temp_Pitch = []
            temp_Roll = []
            temp_Yaw = []

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



P = 1600
generate_map(GPS_Lat_display[10:P],GPS_Lng_display[10:P], AHR_Yaw_display[P])