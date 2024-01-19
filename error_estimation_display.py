import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_excel_data(folder_path):
    # Créer une liste pour stocker les données de chaque fichier
    data_list = []

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        # Vérifier si le fichier est un fichier Excel
        if filename.endswith('.xls') or filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)

            # Lire le fichier Excel en utilisant pandas
            df = pd.read_excel(file_path)

            # Extraire les deux premières colonnes et les ajouter à la liste de données
            data_list.append(df.iloc[:, :2])

    # Vérifier s'il y a des données à afficher
    if len(data_list) == 0:
        print("Aucun fichier Excel valide trouvé dans le dossier.")
        return

    # Concaténer les données en un seul DataFrame
    combined_data = pd.concat(data_list)

    # Créer un graphique à partir des données
    plt.figure(figsize=(10, 6))
    plt.scatter(combined_data.iloc[:, 0], combined_data.iloc[:, 1], marker='o', s=50)
    plt.xlabel("Première colonne")
    plt.ylabel("Deuxième colonne")
    plt.title("Graphique des deux premières colonnes des fichiers Excel")
    plt.grid(True)
    plt.show()

# Remplacez "chemin_du_dossier" par le chemin absolu de votre dossier contenant les fichiers Excel
folder_path = r"C:\Users\edoua\Desktop\Wind Effect\erreurs"
plot_excel_data(folder_path)
