from correction_wf_to_inertial import *
import pandas as pd


import pandas as pd

def save_results_to_excel(errors, time_waypoints, WS, output_file):
    # Créer un dictionnaire avec les données à enregistrer
    data = {
        'Error': errors,
        'Time_Waypoint': time_waypoints,
        'W_speed': WS
    }

    # Créer un DataFrame pandas à partir du dictionnaire
    df = pd.DataFrame(data)

    # Enregistrer le DataFrame dans un fichier Excel
    df.to_excel(output_file, index=False)


def calculate_error(k,N,W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, W_speed, logs_path, time_detection, V_a, Time_Waypoint):

    wf_x, wf_y, xw, yw = process_flight_and_plot(W_speed[k], N, W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path[k], time_detection[k], V_a)

    X_Plume, Y_Plume, X_Trajectoire, Y_Trajectoire, Temps, Distance = process_flight_data(W_speed[k], W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, logs_path[k], time_detection[k])


    waypoints_x, waypoints_y = [],[]

    for i in range(len(Time_Waypoint)):
        if i == 0:
            indice = 0
        else:
            indice = Temps.index(np.around(Time_Waypoint[i]*1e-3))
        waypoints_x.append(xw[indice])
        waypoints_y.append(yw[indice])

    erreur = []
    for i in range(len(waypoints_x)):
        erreur.append(np.sqrt((waypoints_x[i] - wf_x[i])**2 + (waypoints_y[i] - wf_y[i])**2))

    """
    plt.plot(waypoints_x, waypoints_y)
    plt.plot(wf_x, wf_y)
    plt.scatter(wf_x, wf_y)
    plt.scatter(waypoints_x, waypoints_y)
    """

    for i in range(len(Time_Waypoint)):

        Time_Waypoint[i] = (Time_Waypoint[i] - time_detection[k])*1e-3

    if k == 0:
        Time_Waypoint = Time_Waypoint[:14]
        erreur = erreur[:14]

    plt.grid()
    plt.plot(Time_Waypoint, erreur, label = str(W_speed[k]) + '$m.s^{-1}$')
    #plt.scatter(Time_Waypoint, erreur)
    plt.legend(fontsize = 10)
    plt.grid()


    plt.xlabel("Flight Time (s)", fontsize =15)
    plt.ylabel("$Error_{wind \: frame}(m)$", fontsize =15)


    WS = [W_speed[k] for _ in range(len(Time_Waypoint))]
    output_file = r"C:\Users\edoua\Desktop\Wind Effect\erreurs"+ '\\' + str(W_speed[k]) + '.xlsx'
    save_results_to_excel(erreur, Time_Waypoint, WS, output_file)

    return Time_Waypoint, erreur


def extract_columns(input_file, sheet_name, columns_to_extract):

    # Charger le fichier XLSX dans un DataFrame pandas
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    print(df.columns)

    # Extraire les colonnes spécifiées
    extracted_data = df[columns_to_extract]

    return extracted_data


if __name__ == '__main__':

    N=12
    W_direction = 20
    d_P1P2 = 10 * 1e3
    x_plane = -2500
    y_plane = -2500
    Yaw_Plane = 20
    W_speed   = [1,5,7.5,10,12.5,15]
    logs_path = [r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-1-842803.BIN-12889899.mat",r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-5-853904.BIN-3758576.mat", r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-7_5-961403.BIN-3055119.mat", r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-10-1479503.BIN-3355031.mat",r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-12_5-2077043.BIN-2928779.mat", r"C:\Users\edoua\Desktop\Wind Effect\2500-2500-15-1436383.BIN-2986719.mat"]
    time_detection = [842803, 853904, 961403 ,1479503, 2077043 ,1436383]
    V_a = 21.9


    input_file = r"C:\Users\edoua\Desktop\Wind Effect\time_wp.xlsx"
    sheet_name = 'Feuil1'  # Remplacez par le nom de votre feuille
    columns_to_extract = [1,5, 7.5 ,10,12.5,15]
    TP = extract_columns(input_file, sheet_name, columns_to_extract)


    for k in range(len(W_speed)):
        Time_Waypoint, erreur = calculate_error(k,N,W_direction, d_P1P2, x_plane, y_plane, Yaw_Plane, W_speed, logs_path, time_detection, V_a, TP[W_speed[k]])

    plt.grid()


    plt.show()





