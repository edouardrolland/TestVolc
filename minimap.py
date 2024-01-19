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
from distance_calculation import convert_coordinates_ll2xy

import io
from PIL import Image
import pdfkit
import imgkit
import math
import plume_object as p

from Route_E_CPI import *



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



def generate_map(GPS_Lat_display, GPS_Lng_display, angle, map, index, wind_angle, W_speed):

    center_point = [14.4562, -90.8988] #to center the map
    map = folium.Map(location=center_point, zoom_start=14.4, tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community", prefer_canvas=True,)

    ############ Fixed Icons ##########

    marker_summit = folium.Marker(
        location=[lat_fuego, lgn_fuego],
        icon=folium.CustomIcon(icon_image= r"C:/Users/edoua/Documents/Birse/Bristol/MSc Thesis/MiniMap/summit.png", icon_size=(20, 20)), popup='Summit')
    marker_summit.add_to(map)

    marker_start = folium.Marker(
        location=(14.4322662, -90.9349201),
        icon=folium.CustomIcon(icon_image= r"C:/Users/edoua/Documents/Birse/Bristol/MSc Thesis/MiniMap/start_icon.png", icon_size=(23, 23)))
    marker_start.add_to(map)

    coordinates = list(zip(GPS_Lat_display, GPS_Lng_display))

    print(GPS_Lat_display[-1], GPS_Lng_display[-1])

    if coordinates == []:
        return None

    line = folium.PolyLine(
        locations=coordinates,
        color='red',
        weight=2,
        opacity=1
    )
    line.add_to(map) #### plot trajectory

    ############ Moving Airplane ##########
    icon_path = r"C:/Users/edoua/Documents/Birse/Bristol/MSc Thesis/MiniMap/skywalker_icon.png"
    icon = Image.open(icon_path)
    rotated_icon = icon.rotate(-angle, expand=True)
    rotated_icon_path = 'C:/Users/edoua/Documents/Birse/Bristol/MSc Thesis/MiniMap/icone_pivotee.png'
    rotated_icon.save(rotated_icon_path)
    (x,y) = rotated_icon.size
    # Créer un marqueur avec l'icône pivotée
    marker = folium.Marker(
        location=coordinates[-1],
        icon=folium.CustomIcon(icon_image=rotated_icon_path, icon_size=(x/60, y/60))
    )
    marker.add_to(map)

    ############ Wind Heading display ######

    ############ Static Part ############
    # compath_path_background = r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\MiniMap\compass_background.png"
    # compathback_icon = Image.open(compath_path_background)
    # (x,y) = compathback_icon.size
    # print(coordinates[-1])
    # compath_background = folium.Marker(
    #     location=(14.48, -90.84),
    #     icon=folium.CustomIcon(icon_image=compath_path_background, icon_size=(x/10, y/10))
    # )
    # compath_background.add_to(map)


    # """
    # ############ Rotating Part ############
    # compath_path_needle = r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\MiniMap\compass_arrow.png"
    # needle_icon = Image.open(compath_path_needle)
    # rotated_needle = needle_icon.rotate(-wind_angle, expand=True)
    # rotated_needle_path = 'C:/Users/edoua/Documents/Birse/Bristol/MSc Thesis/MiniMap/needle_pivotee.png'
    # rotated_needle.save(rotated_needle_path)
    # (x,y) = rotated_needle.size
    # # Créer un marqueur avec l'icône pivotée
    # needle = folium.Marker(
    #     location=(14.47797, -90.84),
    #     icon=folium.CustomIcon(icon_image=rotated_needle_path, icon_size=(x/10, y/10))
    # )
    #
    # needle.add_to(map)
    # #legend
    # legend_path = r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\MiniMap\wind_direction.png"
    # legend_icon = Image.open(legend_path)
    # (x,y) = legend_icon.size
    #
    # legend = folium.Marker(
    #     location=(14.47, -90.84),
    #     icon=folium.CustomIcon(icon_image=legend_path, icon_size=(x/10, y/10))
    # )
    # legend.add_to(map)
    # """
    ############# Display Plume Test ########

    plume = p.Plume(t_start=0, W_speed=W_speed, W_direction=wind_angle)
    plume.trajectory(100)
    trajectory = folium.PolyLine(plume.trajectory(100), color="blue", weight=1.5, opacity=1, dash_array='10').add_to(map)

    #plume_position = plume.where_geo(0)
    #circle = folium.CircleMarker(location=plume_position, radius=5, popup='Plume Predictor', line_color='#3186cc', fill_color='#3186cc')
    #circle.add_to(map)

    d_P1P2 = 5*1e3
    N = 12
    x_plane, y_plane = convert_coordinates_ll2xy(GPS_Lat_display[-1],  GPS_Lng_display[-1])
    print(x_plane, y_plane)

    plume = p.Plume(t_start=0, W_speed=W_speed, W_direction=wind_angle)
    plume_position = plume.where_meters(0)
    P = plume_position
    x_c, y_c, d_r = E_wind_frame(wind_angle, W_speed, d_P1P2, 22, P)
    x_point, y_point = extract_circle_points(x_c, y_c, d_r, N, wind_angle)
    x_inertial, y_inertial, Time, lat, long, wf_x, wf_y = cpi_trajectory(d_P1P2, 22, W_speed, 12, x_point, y_point, 0, x_plane, y_plane, Yaw, wind_angle)



    coordinates = list(zip(lat, long))

    E_cpi_trajectory = folium.PolyLine(coordinates, color="yellow", weight=1.5, opacity=1).add_to(map)

    for k in range(len(coordinates)):
        if k == 0:
            None
        else:
            folium.Marker(coordinates[k], popup='Waypoint n° ' + str(k)).add_to(map)

    map.save(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\TEST_1\PLUME"+'\\trajectoire_' + str(index) + '.html')


if __name__ == "__main__":

    k = 1400
    index_departure = 20
    wind_angle = 20
    W_speed = 10
    Yaw = AHR_Yaw_display[k]
    generate_map(GPS_Lat_display[index_departure:k],GPS_Lng_display[index_departure:k], Yaw, map,k,wind_angle, W_speed)






