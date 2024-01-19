import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import plume_object as p
from distance_calculation import calculate_distance, convert_coordinates_xy2ll
from dubins_path_planner import *



def check_flight_zone(x_inertial, y_inertial):
    for k in range(len(x_inertial)):
        if x_inertial[k] > 0 or y_inertial[k] > 0:
            if x_inertial[k]**2 + y_inertial[k]**2 >= (2*1e3)**2:
                return False, k
        else:
            if x_inertial[k]**2 + y_inertial[k]**2 >= (20*1e3)**2:
                return False
    return True

def generate_approach_path(Px, Py, x_plane, y_plane, Yaw_Plane, W_direction):

    start_x = float(x_plane)  # [m]
    start_y = float(y_plane)  # [m]
    start_yaw = np.deg2rad(90 - Yaw_Plane)  # [rad]
    end_x = Px  # [m]
    end_y = Px  # [m]
    end_yaw = np.deg2rad(90 - W_direction + 180)  # [rad]
    curvature = 0.2e-2
    path_x, path_y, path_yaw, mode, lenghts = plan_dubins_path(
            start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)

    return path_x, path_y, lenghts



def extract_approach_points(path_x, path_y, lenghts):
    waypoint_x, waypoint_y = [], []

    for k in range(len(path_x)):
        if k%15 == 0:
            waypoint_x.append(path_x[k])
            waypoint_y.append(path_y[k])

    return waypoint_x, waypoint_y

def E_wind_frame(Theta_W, V_W, d_P1P2, V_a, P):
    (Px, Py) = P
    d_r = d_P1P2 * V_a / (2 * np.pi * V_W)
    x_c = Px - d_r * np.cos(np.deg2rad(Theta_W))
    y_c = Py + d_r * np.sin(np.deg2rad(Theta_W))
    return x_c, y_c, d_r


def extract_circle_points(x, y, r, N, Theta_W):
    x_point, y_point = [], []
    theta = np.linspace(-Theta_W, -Theta_W - 360, N + 1)
    for k in range(len(theta)):
        x_point.append(x + r * np.cos(np.deg2rad(theta[k])))
        y_point.append(y + r * np.sin(np.deg2rad(theta[k])))
    return x_point, y_point


def cpi_trajectory(d_P1P2, V_a, V_W, plume, N, x_waypoints_wf, y_waypoints_wf, t_start, x_plane, y_plane):

    x_inertial, y_inertial = [], []
    Px, Py = x_waypoints_wf[0], y_waypoints_wf[0]
    path_x, path_y, lenghts = generate_approach_path(Px, Py, x_plane, y_plane, Yaw_Plane, W_direction)

    way_x, way_y = extract_approach_points(path_x, path_y, lenghts)
    print(len(path_x))


    t_flight_time = 0

    for k in range(len(way_x)):
        t_flight_time += np.sum(lenghts)/(len(path_x) * V_a)
        x_inertial.append(way_x[k] + plume.where_meters(t_start + t_flight_time)[0])
        y_inertial.append(way_y[k] + plume.where_meters(t_start + t_flight_time)[1])

    for k in range(len(x_waypoints_wf)):
        if k == 0:
            t_flight_time = np.sum(lenghts)/ V_a
        else:
            t_flight_time += 2 * np.pi * d_r / (N * V_a)


        try:
            x_inertial.append(x_waypoints_wf[k] + plume.where_meters(t_start + t_flight_time)[0])
            y_inertial.append(y_waypoints_wf[k] + plume.where_meters(t_start + t_flight_time)[1])

        except:
            print(" WARNING Plume Outside the flight zone ")

    plt.figure()
    plt.plot(x_inertial, y_inertial)
    plt.scatter(x_inertial, y_inertial)
    plt.axis('equal')
    plt.show()

    return x_inertial, y_inertial


def E_get_cpi(W_speed, W_direction, d_P1P2, V_a, N_waypoints, t_start, t, x_plane, y_plane):

    plume = p.Plume(t_start=t_start, W_speed=W_speed, W_direction=W_direction)
    plume_position = plume.where_meters(0)

    x_c, y_c, d_r = E_wind_frame(W_direction, W_speed, d_P1P2, V_a,
                                 plume_position)  # we get the parameters associated to the plume circle in the wind frame
    x_waypoints_wf, y_waypoints_wf = extract_circle_points(x_c, y_c, d_r, N_waypoints, W_direction)

    x_inertial, y_inertial = cpi_trajectory(d_P1P2, V_a, W_speed, plume, N_waypoints, x_waypoints_wf, y_waypoints_wf, t_start, x_plane, y_plane )

    if check_flight_zone(x_inertial, y_inertial) == False:
        print("The generated waypoints are outside the flight zone")
        return False

    lat, long = [], []
    for k in range(len(x_inertial)):
        conversion = convert_coordinates_xy2ll(14.4747, -90.8806, x_inertial[k], y_inertial[k])
        lat.append(conversion[0])
        long.append(conversion[1])

    return lat, long


if __name__ == "__main__":


    """
    We define the various useful constants
    """

    W_speed = 12
    W_direction = 20
    d_P1P2 = 15 * 1e3
    V_a = 25
    N = 12
    Yaw_Plane = 30
    x_plane, y_plane = -2500, 0


    """       Define the plume object        """

    plume = p.Plume(t_start=0, W_speed=W_speed, W_direction=W_direction)
    plume_position = plume.where_meters(0)
    P = plume_position
    x_c, y_c, d_r = E_wind_frame(W_direction, W_speed, d_P1P2, V_a, P)
    x_point, y_point = extract_circle_points(x_c, y_c, d_r, N, W_direction)

    cpi_trajectory(d_P1P2, V_a, W_speed, plume, N, x_point, y_point, 0, x_plane, y_plane)




