import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
logs_path = r"D:\Volcanology Videos\2019-03-25_1_LeiaPlumeAxis\Flight Logs\12.BIN-1610065.mat"
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

"""""""""""""""""""""""""""
    Fuego coordinates
"""""""""""""""""""""""""""

###### Extract GPS Data, IMU_Data, ARH Data ######

WIND = logs_data['NKF2']

WIND_Time = WIND[:, 1]
WIND_N = WIND[:, 6]
WIND_E = WIND[:, 7]

WIND_Time = WIND_Time * 1e-6

Norme = (np.array(WIND_N)**2 + np.array(WIND_E)**2)**0.5

for k in range(len(Norme)):
    if np.around(Norme[k]) != 0.0:
        index_departure = k
        time_departure = WIND_Time[k]
        break

for k in range(len(GPS_Alt)):
    if np.abs(GPS_Alt[k] - GPS_Alt[k+1]) > 0.7:
        index_departure_GPS = k
        time_departure_GPS = GPS_Time[k]
        break


# Création du graphique et des axes
fig, ax1 = plt.subplots()

# Tracé de la première courbe
ax1.plot(GPS_Time[index_departure_GPS:,] - time_departure ,GPS_Alt[index_departure_GPS:,], 'b-')
ax1.set_xlabel('Flight Time (s)', fontsize = 15)
ax1.set_ylabel('Altitude (m)', color='b', fontsize = 15)
ax1.tick_params('y', colors='b')

# Création d'un deuxième axe partageant le même x
ax2 = ax1.twinx()

# Tracé de la deuxième courbe
ax2.plot(WIND_Time[index_departure:,] - time_departure, Norme[index_departure:,], 'r-')
ax2.set_ylabel('Wind Speed ($m.s^{-1}$)', color='r', fontsize = 15)
ax2.tick_params('y', colors='r')

# Affichage du graphique
plt.grid()
plt.show()

print(np.mean(Norme[index_departure:,]))


# Création de la figure et de l'axe
fig, ax = plt.subplots()

# Création du boxplot
bp = ax.boxplot(Norme[index_departure:,])

# Personnalisation des axes et des titres
ax.set_xticklabels(['Wind'])
ax.set_xlabel('Données')
ax.set_ylabel('Valeurs')

# Affichage du boxplot
plt.show()

