import numpy as np
import matplotlib.pyplot as plt

# Générer des données fictives pour quatre groupes
Erreur_Point_1 = [54.31528442, 10.05442759, 77.06588295, 46.657921, 39.14247024, 127.1120678, 125.9380803, 125.3101731, 118.6823159]
Erreur_Point_2 = [841.415289, 833.2428496, 827.5158646, 845.4819144, 845.8758642, 856.8775353, 853.7557825, 821.0867033, 882.9726]

# Création de la figure et des axes
fig, ax1 = plt.subplots()

# Création de la première courbe avec l'échelle de gauche
box1 = ax1.boxplot(Erreur_Point_1, positions=[1], patch_artist=True)

# Création de la deuxième échelle pour la deuxième courbe
ax2 = ax1.twinx()
box2 = ax2.boxplot(Erreur_Point_2, positions=[2], patch_artist=True)

# Couleurs des boxplots
box_colors = ['#1f77b4', '#2ca02c']
for box, color in zip([box1, box2], box_colors):
    for patch in box['boxes']:
        patch.set_facecolor(color)

# Configuration des étiquettes de l'axe x
ax1.set_xticks([1, 2])
ax1.set_xticklabels(['$d_1$', '$d_2$'], fontsize = 25)

# Coloration des axes verticaux et agrandissement des numéros de légende
ax1.tick_params(axis='y', colors=box_colors[0], labelsize=20)
ax2.tick_params(axis='y', colors=box_colors[1], labelsize=20)

ax1.set_ylabel('Minimum Distance (m)', fontsize = 15)

"""
plt.figure()
data = [22.93, 22.84, 22.61, 23.18, 23.13, 23.78, 23.49, 23.40, 23.47]
plt.boxplot(data, labels=[''])
plt.ylabel('%', fontsize = 15)
# Agrandir la taille des chiffres de l'échelle verticale
plt.tick_params(axis='y', labelsize=20)
"""

plt.figure()

import matplotlib.pyplot as plt

# Données fournies
vitesse = [1, 5, 7.5, 10, 12.5, 15]
d_1 = [4.00979219, 57.2704784, 189.350834, 346.969351, 526.418294, 1040.1078]
d_2 = [568.291331, 600.888881, 643.085092, 1025.13087, 1445.08206, 2995.9477]

plt.scatter(vitesse, d_1, label='d_1')
plt.scatter(vitesse, d_2, label='d_2', color='orange')
plt.xlabel('Vitesse')
plt.ylabel('Distance')
plt.legend()
plt.show()

plt.show()


