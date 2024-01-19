import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def interceptor(x_plume, y_plume, x_plane, y_plane, W_speed, V_plane, W_direction):

    theta_J = np.arctan(np.abs(x_plane - x_plume)/np.abs(y_plane - y_plume))
    theta_D = np.pi - np.radians(W_direction) - theta_J
    D = np.sqrt((x_plane - x_plume)**2 + (y_plane - y_plume)**2)
    A = (V_plane**2 - W_speed**2)
    B = 2*D*W_speed*np.cos(theta_D)
    C = -D**2
    coefficients = [A, B, C]
    racines = np.roots(coefficients)

    return racines[0]




