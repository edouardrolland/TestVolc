import numpy as np
import matplotlib.pyplot as plt
import plume_object as p
from distance_calculation import calculate_distance, convert_coordinates_xy2ll, distance_entre_deux_points
from dubins_path_planner import *
import matplotlib.animation as animation
#from error_regression import *
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.animation import FuncAnimation



def error_prediction(t, W_speed): #C'est pas très optimal mais bon c'est certain que c'est correct

    columns_to_extract = ['Time_Waypoint', 'W_speed', 'Error']
    data = extract_columns(r"C:\Users\edoua\Desktop\Wind Effect\erreurs\global.xlsx", 'Feuil1', columns_to_extract)

    x = data[['Time_Waypoint', 'W_speed']]
    y = data['Error']
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    x_poly = poly_features.fit_transform(x.values)

    # Créer le modèle de régression linéaire pour la régression polynomiale avec le meilleur degré
    regression_poly = LinearRegression()
    regression_poly.fit(x_poly, y)
    new_x_poly = poly_features.transform([[t, W_speed]])
    error = regression_poly.predict(new_x_poly)

    return error


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
        if k%2 == 0:
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

def cpi_trajectory(d_P1P2, V_a, V_W, N, x_waypoints_wf, y_waypoints_wf, t_start, x_plane, y_plane, Yaw_Plane, W_direction):

    x_inertial, y_inertial = [], []
    Px, Py = 0,0
    path_x, path_y, lenghts = generate_approach_path(Px, Py, x_plane, y_plane, Yaw_Plane, W_direction)

    way_x, way_y = extract_approach_points(path_x, path_y, lenghts)
    t_flight_time = 0

    wf_x = way_x + x_waypoints_wf
    wf_y = way_y + y_waypoints_wf

    Temps = [0]
    t_flight_time = 0
    for k in range(len(wf_x)):
        if k == 0:
            None
        else:
            d = distance_entre_deux_points(wf_x[k-1],wf_y[k-1], wf_x[k], wf_y[k])
            t = d/(V_a)
            t_flight_time = t_flight_time + t
            Temps.append(t_flight_time)

    print("le temps de vol est " + str(Temps[-1]))

    plume_simu = p.Plume(t_start=0, W_speed=V_W, W_direction=W_direction)
    x_inertial, y_inertial = [], []

    for k in range(len(Temps)):
        x_inertial.append(wf_x[k] + plume_simu.where_meters(Temps[k])[0])
        y_inertial.append(wf_y[k] + plume_simu.where_meters(Temps[k])[1])

    lat, long = [], []

    for k in range(len(x_inertial)):
        conversion = convert_coordinates_xy2ll(14.4747, -90.8806, x_inertial[k], y_inertial[k])
        lat.append(conversion[0])
        long.append(conversion[1])

    return x_inertial, y_inertial, Temps, lat, long, wf_x, wf_y


def cpi_trajectory_corrected(d_P1P2, V_a, V_W, N, x_waypoints_wf, y_waypoints_wf, t_start, x_plane, y_plane, Yaw_Plane, W_direction):

    x_inertial, y_inertial = [], []
    Px, Py = 0,0
    path_x, path_y, lenghts = generate_approach_path(Px, Py, x_plane, y_plane, Yaw_Plane, W_direction)

    way_x, way_y = extract_approach_points(path_x, path_y, lenghts)
    t_flight_time = 0

    wf_x = way_x + x_waypoints_wf
    wf_y = way_y + y_waypoints_wf

    if __name__ == "__main__":
        plt.plot(wf_x, wf_y)
        plt.scatter(wf_x, wf_y)
        plt.grid()
        plt.axis('equal')
        plt.show()

    Temps = [0]
    t_flight_time = 0
    for k in range(len(wf_x)):
        if k == 0:
            None
        else:
            d = distance_entre_deux_points(wf_x[k-1],wf_y[k-1], wf_x[k], wf_y[k])
            t = d/(V_a)
            t_flight_time = t_flight_time + t
            Temps.append(t_flight_time)

    print("le temps de vol est " + str(Temps[-1]))

    #################### NOW WE CONSIDER THE CORRECTION ########################

    coefficients = [ 0.00000000e+00, -1.15552844e-01, -6.47696723e+01,  9.21823378e-06, 3.39808803e-01,  6.05983447e+00]

    intercept = 104.38810159421189

    def erreur(coefficients,intercept,Wind_Speed,t):
        A = intercept + coefficients[1]*t + coefficients[2]*Wind_Speed + coefficients[3]*t**2 + coefficients[4]*t*Wind_Speed + coefficients[5]*Wind_Speed**2
        return A

    plume_simu = p.Plume(t_start=0, W_speed=V_W, W_direction=W_direction)
    x_inertial, y_inertial = [], []

    for k in range(len(Temps)):
        error = erreur(coefficients,intercept,V_W,Temps[k])
        x_inertial.append(wf_x[k] + plume_simu.where_meters(Temps[k])[0] + np.sin(np.deg2rad(W_direction)) * error)
        y_inertial.append(wf_y[k] + plume_simu.where_meters(Temps[k])[1] + np.cos(np.deg2rad(W_direction)) * error)

    lat, long = [], []

    for k in range(len(x_inertial)):
        conversion = convert_coordinates_xy2ll(14.4747, -90.8806, x_inertial[k], y_inertial[k])
        lat.append(conversion[0])
        long.append(conversion[1])

    return x_inertial, y_inertial, Temps, lat, long, wf_x, wf_y


if __name__ == "__main__":

    """
    We define the various useful constants
    """

    W_speed = 10
    W_direction = 20
    d_P1P2 = 5 * 1e3
    V_a = 22
    N = 12
    x_plane, y_plane = -2000, -500
    Yaw_Plane = 15

    """       Define the plume object        """

    plume = p.Plume(t_start=0, W_speed=W_speed, W_direction=W_direction)
    plume_position = plume.where_meters(0)
    P = plume_position
    x_c, y_c, d_r = E_wind_frame(W_direction, W_speed, d_P1P2, V_a, P)
    x_point, y_point = extract_circle_points(x_c, y_c, d_r, N, W_direction)

    x_inertial, y_inertial, Time, lat, long, wf_x, wf_y = cpi_trajectory(d_P1P2, V_a, W_speed, N, x_point, y_point, 0, x_plane, y_plane, Yaw_Plane, W_direction)
    fig, ax = plt.subplots()
    cercle = plt.Circle((x_c, y_c), d_r, color='orange', fill=False, label='CPI Trajectory')
    ax.add_artist(cercle)


    # Ajouter des annotations avec des flèches de plus grande taille
    for i in range(0, len(path_x), 20):
        plt.annotate('', xy=(path_x[i+1 + 5], path_y[i+1 +5]), xytext=(path_x[i+5], path_y[i +5]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#1f77b4", mutation_scale=20))
    #plt.scatter(path_x, path_y)

    theta = np.linspace(0, 2 * np.pi, 100)  # Création d'un tableau de valeurs allant de 0 à 2*pi
    x_cercle = x_c + d_r * np.cos(theta)  # Coordonnées x des points sur le cercle
    y_cercle = y_c + d_r * np.sin(theta)  # Coordonnées y des points sur le cercle
    #plt.scatter(x_cercle, y_cercle)
    x_cercle = x_cercle[::-1]
    y_cercle = y_cercle[::-1]

    for i in range(0, len(x_cercle), 22):
        plt.annotate('', xy=(x_cercle[i+1 + 10], y_cercle[i+1 +10]), xytext=(x_cercle[i+10], y_cercle[i +10]),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#ff7f0e", mutation_scale=20))


    plt.scatter(wf_x[:7], wf_y[:7], label = 'Approach Trajectory Waypoints')
    plt.scatter(wf_x[7:], wf_y[7:], label = 'CPI Trajectory Waypoints')
    plt.scatter(x_c, y_c, c='k')

    t = np.linspace(-2, 2, 100)
    y = np.sin(t)
    # return the handle of the line
    line = plt.plot(path_x, path_y)[0]

    x_orange, y_orange = wf_x[13], wf_y[13]

    # Calculate the midpoint
    mid_x = (x_orange + x_c) / 2
    mid_y = (y_orange + y_c) / 2

    # Add label at the midpoint
    plt.text(mid_x+220, mid_y, '$R_{wind}$', fontsize=15, ha='center', va='center')


    # Dessiner une double flèche entre le point noir et le point orange
    plt.annotate('', xy=(x_orange, y_orange), xytext=(x_c, y_c),
             arrowprops=dict(arrowstyle="<->", color="k"))
    plt.xlabel('$x_{wind~frame}(m)$',fontsize=15)
    plt.ylabel('$y_{wind~frame}(m)$',fontsize=15)
    # Point de départ de la flèche
    x_start = wf_x[0]
    y_start = wf_y[0]

    length = 750
    # Conversion de l'angle en radians
    angle_radians = np.radians(Yaw_Plane)

    # Calcul des composantes de la flèche
    dx = length * np.sin(angle_radians)
    dy = length * np.cos(angle_radians)

    # Longueur de la flèche
    arrow_length = 750


    # Ajouter une flèche verticale

    plt.arrow(x_start, y_start, dx, dy, head_width=50, head_length=50, fc='red', ec='red')

    plt.text(x_start, y_start - 200, "$Drone(t_0)$", fontsize=12, ha='center')



    x_start = wf_x[6]
    y_start = wf_y[6]

    # Longueur de la flèche
    length = 750

    # Angle de la flèche en degrés (200 degrés = 180 + 20, en sens horaire depuis le nord)
    angle_degrees = -20
    # Conversion de l'angle en radians
    angle_radians = np.radians(angle_degrees)

    # Calcul des composantes de la flèche
    dx = length * np.sin(angle_radians)
    dy = length * np.cos(angle_radians)

    # Dessiner la flèche rouge
    plt.arrow(x_start, y_start, dx, -dy, head_width=50, head_length=50, fc='red', ec='red')
    plt.text(x_start+200, y_start, "$P_{1,2}$", fontsize=12, ha='center')
    plt.text(x_start +200, y_start-500, "Wind \n Vector", fontsize=12, ha='center')

    plt.text(-800, y_start, "$\theta_{wind}$", fontsize=14, ha='center')
    plt.text(-900, y_start-1000, "$\\theta_{drone}$", fontsize=14, ha='center')

    plt.axis('equal')
    plt.legend(fontsize=12, loc = 'upper right')
    plt.grid()
    plt.show()







    #
    # plt.figure()
    # plt.plot(x_inertial, y_inertial, label = 'corrected')
    #
    # x_inertial, y_inertial, Time, lat, long, wf_x, wf_y = cpi_trajectory(d_P1P2, V_a, W_speed, N, x_point, y_point, 0, x_plane, y_plane, Yaw_Plane, W_direction)
    # plt.plot(x_inertial, y_inertial, label = 'initial')
    # plt.legend()
    # plume_3 = p.Plume(t_start=0, W_speed=W_speed, W_direction=W_direction)
    # X_Plume = []
    # Y_Plume = []
    # for k in range(len(Time)):
    #     position = plume_3.where_meters(Time[k])
    #     X_Plume.append(position[0])
    #     Y_Plume.append(position[1])
    #
    # plt.scatter(X_Plume, Y_Plume)
    #
    # # Remplacez ces listes par vos coordonnées X1, Y1, X2 et Y2 ainsi que les temps
    # X1 = X_Plume
    # Y1 = Y_Plume
    #
    # X2 = x_inertial
    # Y2 = y_inertial
    #
    # temps = Time  # Assurez-vous que la longueur de temps correspond à celle de X1, Y1, X2 et Y2
    #
    # fig, ax = plt.subplots()
    # sc1 = ax.scatter([], [], c='b', marker='o', label='Plume')
    # sc2 = ax.scatter([], [], c='r', marker='o', label='Plane')
    # line1, = ax.plot([], [], c='b', linestyle='dashed')
    # line2, = ax.plot([], [], c='r', linestyle='dashed')
    #
    # def init():
    #     #ax.axis('equal')
    #     ax.set_xlim(min(min(X1), min(X2)) - 2000, max(max(X1), max(X2)) + 2000)
    #     ax.set_ylim(min(min(Y1), min(Y2)) - 2000, max(max(Y1), max(Y2)) + 2000)
    #     # Ajout du repère orthonormé
    #     ax.legend()
    #
    #
    #     return sc1, sc2, line1, line2
    #
    # def update(frame):
    #     sc1.set_offsets(np.column_stack((X1[:frame+1], Y1[:frame+1])))
    #     sc2.set_offsets(np.column_stack((X2[:frame+1], Y2[:frame+1])))
    #
    #     line1.set_data(X1[:frame+1], Y1[:frame+1])
    #     line2.set_data(X2[:frame+1], Y2[:frame+1])
    #
    #     return sc1, sc2, line1, line2
    #
    # ani = FuncAnimation(fig, update, frames=len(temps), init_func=init, blit=True)
    #
    #
    # # saving to m4 using ffmpeg writer
    # writergif = animation.PillowWriter(fps=5)
    # ani.save(r"C:\Users\edoua\Desktop\theory.gif", writer=writergif)
    # print("coucou")


    plt.show()






