import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import datetime
from scipy.io import loadmat
from distance_calculation import convert_coordinates_ll2xy
import plume_object as p
from Route_E_CPI import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def process_flight_data(W_speed, W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path, time_detection):
    lat_fuego = 14.4747
    lgn_fuego = -90.8806
    N_waypoints = 12
    t_start = 0
    N = 12

    # Load telemetry data from the specified logs_path
    logs_data = loadmat(logs_path)

    # Extract GPS data from the logs_data
    GPS = logs_data['GPS_0']
    GPS_Time = GPS[:, 1]
    GPS_Lat = GPS[:, 8]
    GPS_Lng = GPS[:, 9]
    GPS_Alt = GPS[:, 10]
    GPS_Spd = GPS[:, 11]

    GPS_Time_display = []
    GPS_Lat_display = []
    GPS_Lng_display = []
    GPS_Alt_display = []
    GPS_Spd_display = []

    GPS_Time = GPS_Time * 1e-6

    # Sequential smoothing for GPS data
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

    # Clean Data
    time_detection = np.around(time_detection * 1e-3 )
    print(time_detection)
    index_detection = GPS_Time_display.index(time_detection)

    X_Trajectoire, Y_Trajectoire = [], []

    for i in range(len(GPS_Time_display)):
        if i > index_detection:
            x, y = convert_coordinates_ll2xy(GPS_Lat_display[i], GPS_Lng_display[i])
            X_Trajectoire.append(x)
            Y_Trajectoire.append(y)

    def plume_trajectoire(W_speed, time_detection):
        W_speed = W_speed
        plume_2 = p.Plume(t_start=time_detection, W_speed=W_speed, W_direction=W_direction)
        X_Plume, Y_Plume = [], []
        Temps = []
        for k in range(len(GPS_Time_display)):
            if GPS_Time_display[k] > time_detection:
                A = plume_2.where_meters(GPS_Time_display[k])
                X_Plume.append(A[0])
                Y_Plume.append(A[1])
                Temps.append(GPS_Time_display[k])
        return X_Plume, Y_Plume, Temps

    def distance(X_Plume, Y_Plume, Temps, X_Trajectoire, Y_Trajectoire):
        res = []
        for k in range(len(Temps)):
            res.append(np.sqrt((X_Plume[k] - X_Trajectoire[k]) ** 2 + (Y_Plume[k] - Y_Trajectoire[k]) ** 2))
        return res

    X_Plume, Y_Plume, Temps = plume_trajectoire(W_speed, time_detection)

    Distance = distance(X_Plume, Y_Plume, Temps, X_Trajectoire, Y_Trajectoire)

    return X_Plume, Y_Plume, X_Trajectoire, Y_Trajectoire, Temps, Distance

def plot_animation(X_Plume, Y_Plume, X_Trajectoire, Y_Trajectoire, Temps):
    fig, ax = plt.subplots()
    sc1 = ax.scatter([], [], c='b', marker='o', label='Plume')
    sc2 = ax.scatter([], [], c='r', marker='o', label='AirPlane')
    line1, = ax.plot([], [], c='b', linestyle='dashed')
    line2, = ax.plot([], [], c='r', linestyle='dashed')

    def init():
        ax.set_xlim(min(min(X_Plume), min(X_Trajectoire)) - 1000, max(max(X_Plume), max(X_Trajectoire)) + 1000)
        ax.set_ylim(min(min(Y_Plume), min(Y_Trajectoire)) - 1000, max(max(Y_Plume), max(Y_Trajectoire)) + 1000)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        ax.legend()
        return sc1, sc2, line1, line2

    def update(frame):
        sc1.set_offsets([[X_Plume[frame], Y_Plume[frame]]])
        sc2.set_offsets([[X_Trajectoire[frame], Y_Trajectoire[frame]]])
        line1.set_data(X_Plume[:frame + 1], Y_Plume[:frame + 1])
        line2.set_data(X_Trajectoire[:frame + 1], Y_Trajectoire[:frame + 1])
        return sc1, sc2, line1, line2

    ani = FuncAnimation(fig, update, frames=len(Temps), init_func=init, blit=True, interval=10)

    # saving to m4 using ffmpeg writer
    #writergif = animation.PillowWriter(fps=30)
    #ani.save(r"C:\Users\edoua\Desktop\SITL_true.gif", writer=writergif)
    plt.show()

def plot_distance(Temps, Distance):

    """
    plt.figure()
    plt.plot(Temps, Distance)
    print(trouver_minimums_locaux(Distance))
    plt.show()
    """
def trouver_minimums_locaux(courbe):
    if len(courbe) < 3:
        raise ValueError("La liste doit contenir au moins trois éléments.")

    minimums_locaux = []
    indices = []

    for i in range(1, len(courbe) - 1):
        if courbe[i - 1] > courbe[i] < courbe[i + 1]:
            minimums_locaux.append(courbe[i])
            indices.append(i)
    print('les indices sont' + str(indices))

    return np.sum(minimums_locaux), minimums_locaux

def plot_wind_frame(X_Trajectoire, Y_Trajectoire, Temps, W_speed, W_direction, t_start):
    xw, yw = [], []
    plume_r = p.Plume(t_start=t_start, W_speed=W_speed, W_direction=W_direction)
    for k in range(len(Temps)):
        xw.append(X_Trajectoire[k] - plume_r.where_meters(Temps[k])[0])
        yw.append(Y_Trajectoire[k] - plume_r.where_meters(Temps[k])[1])
    return xw, yw


def process_flight_and_plot(W_speed, N, W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path, time_detection, V_a):
    # The code from your previous script
    X_Plume, Y_Plume, X_Trajectoire, Y_Trajectoire, Temps, Distance = process_flight_data(W_speed, W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path, time_detection)

    #plot_animation(X_Plume, Y_Plume, X_Trajectoire, Y_Trajectoire, Temps)
    #plot_distance(Temps, Distance)


    xw, yw = plot_wind_frame(X_Trajectoire, Y_Trajectoire, Temps, W_speed, W_direction, time_detection*1e-3)


    plume = p.Plume(t_start=0, W_speed=W_speed, W_direction=W_direction)
    plume_position = plume.where_meters(0)
    P = plume_position
    x_c, y_c, d_r = E_wind_frame(W_direction, W_speed, d_P1P2, V_a, P)
    x_point, y_point = extract_circle_points(x_c, y_c, d_r, N, W_direction)

    x_inertial, y_inertial, Time, lat, long, wf_x, wf_y = cpi_trajectory(d_P1P2, V_a, W_speed, N, x_point, y_point, 0, x_plane, y_plane, Yaw_Plane, W_direction)


    return wf_x, wf_y, xw, yw




if __name__ == '__main__':

    N=12
    W_direction = 20
    d_P1P2 = 10 * 1e3
    x_plane = -2500
    y_plane = -2500
    Yaw_Plane = 20
    W_speed   = [10,10]
    logs_path = [r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-10-1479503.BIN-3355031.mat",r"C:\Users\edoua\Desktop\Wind Effect\Modèles corrigés avec la régression\T1-2500-2500-10-1180544.BIN-3434011.mat"]
    time_detection = [1479503, 1180544]
    V_a = 21.9
    plt.axis("equal")
    k = 0
    wf_x wf_y, xw, yw = process_flight_and_plot(W_speed[k], N, W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path[k], time_detection[k], V_a)

    plt.plot(xw, yw,"--", label = 'SITL non corrected')
    plt.plot(wf_x, wf_y, label = "Theory")
    plt.legend()
    k = 1
    wf_x, wf_y, xw, yw = process_flight_and_plot(W_speed[k], N, W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path[k], time_detection[k], V_a)
    plt.plot(xw, yw, label = 'SITL corrected')
    plt.legend()
    plt.show()




