import numpy as np
import matplotlib.pyplot as plt
import plume_object as p
from distance_calculation import calculate_distance, convert_coordinates_xy2ll


def C_wind_frame(Theta_W, V_W, d_P1P2, V_a, P):
    (Px, Py) = P
    d_path = V_a * d_P1P2 / (2 * V_W)

    x_c = Px - d_path * np.cos(np.deg2rad(Theta_W))
    y_c = Py + d_path * np.sin(np.deg2rad(Theta_W))

    return x_c, y_c


def C_get_cpi(W_speed, W_direction, d_P1P2, V_a, t_start, t):
    plume = p.Plume(t_start=t_start, W_speed=W_speed, W_direction=W_direction)
    try:
        plume_position = plume.where_meters(t)
    except:
        print("PLume Outside Flight Zone")
        return False
    x_c, y_c = C_wind_frame(W_direction, W_speed, d_P1P2, V_a, plume_position)
    x_waypoints_wf, y_waypoints_wf = [plume_position[0], x_c, plume_position[0]], [plume_position[1], y_c,
                                                                                   plume_position[1]]
    x_inertial, y_inertial = [], []
    t_flight_time = 0

    for k in range(len(x_waypoints_wf)):

        if k == 0:
            t_flight_time = 0
        else:
            t_flight_time += d_P1P2 / (2 * W_speed)

        try:
            x_inertial.append(x_waypoints_wf[k] + plume.where_meters(t_start + t_flight_time)[0])
            y_inertial.append(y_waypoints_wf[k] + plume.where_meters(t_start + t_flight_time)[1])
        except:
            print('WARNING Plume outside flight zone')
            return False

    print(np.sqrt((x_inertial[0] - x_inertial[1]) ** 2 + (y_inertial[0] - y_inertial[1]) ** 2))
    lat, long = [], []
    for k in range(len(x_inertial)):
        conversion = convert_coordinates_xy2ll(14.4747, -90.8806, x_inertial[k], y_inertial[k])
        lat.append(conversion[0])
        long.append(conversion[1])

    return lat, long


if __name__ == "__main__":
    W_speed = 15
    W_direction = 20
    d_P1P2 = 10 * 1e3
    V_a = 25

    plume = p.Plume(t_start=0, W_speed=W_speed, W_direction=W_direction)

    plume_position = plume.where_meters(10)

    P = plume_position

    Plot_x = P[0] + np.sin(np.deg2rad(W_direction)) * 3e3
    Plot_y = P[1] + np.cos(np.deg2rad(W_direction)) * 3e3

    x_c, y_c = C_wind_frame(W_direction, W_speed, d_P1P2, V_a, P)

    plt.scatter(P[0], P[1])
    plt.plot([P[0], Plot_x], [P[1], Plot_y], 'r-')
    plt.scatter(x_c, y_c)
    plt.axis('equal')
    plt.show()

    lat, long = C_get_cpi(W_speed=12, W_direction=20, d_P1P2=15 * 1e3, V_a=25, t_start=0, t=100)
