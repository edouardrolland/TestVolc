from pymavlink import mavutil
from Route_E_CPI import E_get_cpi
from STIL_change_param import *
import time


################ SET UP THE COORDINATE INTERCEPTION TRAJECTORY #################

W_speed = 5
W_direction = 20
d_P1P2 = 5 * 1e3
V_a = 22
N_waypoints = 12
t_start = 0
t = 0
x_plane = -2500
y_plane = -2500
Yaw_plane = 20

lat_cpi, lgn_cpi = E_get_cpi(W_speed, W_direction, d_P1P2, V_a, N_waypoints, t_start, t, x_plane, y_plane, Yaw_plane)
coordinates = list(zip(lat_cpi, lgn_cpi))

STIL_change_weather(0, W_direction)


the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0)
msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
print(msg)
