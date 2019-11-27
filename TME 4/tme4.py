import numpy as np
from math import *
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-téte
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break
    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )

    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data


# calculation de densité en utilisant les expressions de NumPy
def normale_bidim(x, z, params):
    # on peut utiliser cette fonction lorsque x et z sont les nbs de type "float"
    # et, aussi, lorsque x et z sont de tableaux de type "numpy.ndarray"
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    return np.exp(-(
        ((x - mu_x) / sigma_x) ** 2 -
        2.0 * rho * (x - mu_x) * (z - mu_z) / sigma_x / sigma_z +
        ((z - mu_z) / sigma_z) ** 2
    ) / 2 / (1 - rho ** 2)) / 2 / np.pi / sigma_x / sigma_z / np.sqrt(1 - rho ** 2)


def dessine_1_normale(params, sigma_mult=2.0, grid_dot_num=100, figi=-2):
    # récupération des paramétres
    mu_x, mu_z, sigma_x, sigma_z, rho = params
    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - sigma_mult * sigma_x
    x_max = mu_x + sigma_mult * sigma_x
    z_min = mu_z - sigma_mult * sigma_z
    z_max = mu_z + sigma_mult * sigma_z
    # création de la grille
    x = np.linspace(x_min, x_max, grid_dot_num)
    z = np.linspace(z_min, z_max, grid_dot_num)
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    #
    # Comme on peut calculer des normales en passant directement des np.ndarray à
    # fonction normale_bidim, j'ai commenté cette partie de code où on les calcule par
    # boucles. En fait, l'utilisation des boucles est beaucoup plus chère en temps
    # que l'utilisation des objets et des méthodes de NumPy.
    # C'est pour cette raison que je l'ai commentée.
    # Vous pouvez vérifier que toutes les deux manières de calculation retournent
    # les mêmes résultats en décommentant le code ci-dessous.
    #
    # norm = X.copy()
    # for i in range(x.shape[0]):
    #     for j in range(z.shape[0]):
    #         norm[i, j] = normale_bidim(x[i], z[j], params)
    # print(np.linalg.norm(norm - norm_np))

    norm_np = normale_bidim(X, Z, params)
    # affichage
    fig = plt.figure(figi)
    plt.contour(X, Z, norm_np, cmap=cm.autumn)
    plt.show()


def dessine_normales(data, params, weights, bounds, ax, grid_dot_num=100, alpha=0.5):
    # récupération des paramétres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]
    # on détermine les coordonnées des coins de la figure
    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]
    # création de la grille
    nb_x = nb_z = grid_dot_num
    x = np.linspace(x_min, x_max, nb_x)
    z = np.linspace(z_min, z_max, nb_z)
    X, Z = np.meshgrid(x, z)
    # calcul des normales
    norm0_np = normale_bidim(X, Z, params[0]) * weights[0]
    norm1_np = normale_bidim(X, Z, params[1]) * weights[1]

    # Pour les raisons indiquées à la méthode précédente, j'ai commenté le code ci-dessous
    # Vous pouvez vérifier que toutes les deux manières de calculation retournent
    # les mêmes résultats en décommentant ce code.

    # norm0 = np.zeros((nb_x, nb_z))
    # for j in range(nb_z):
    #     for i in range(nb_x):
    #         norm0[j, i] = normale_bidim(x[i], z[j], params[0])  # * weights[0]
    # norm1 = np.zeros((nb_x, nb_z))
    # for j in range(nb_z):
    #     for i in range(nb_x):
    #         norm1[j, i] = normale_bidim(x[i], z[j], params[1])  # * weights[1]
    # print(np.linalg.norm(norm0 - norm0_np))
    # print(np.linalg.norm(norm1 - norm1_np))

    # affichages des normales et des points du dataset
    ax.contour(X, Z, norm0_np, cmap=cm.winter, alpha=alpha)
    ax.contour(X, Z, norm1_np, cmap=cm.autumn, alpha=alpha)
    for point in data:
        ax.plot(point[0], point[1], 'k+')


# Un pas de l'étape E pour l'algoritme EM
def Q_i(data, current_params, current_weights):
    # On va calculer une valeur Qi en utilisant les expressions NumPy
    x, z = data[:, 0], data[:, 1]                                # récupération de x et z
    params_0, params_1 = current_params[0], current_params[1]    # récuperation de paramètres
    weight_0, weight_1 = current_weights[0], current_weights[1]  # récuperation de poids
    # on calcule alpha0 et alpha1 en passant x, z et paramètres à la fonction normale_bidim
    alpha_0 = weight_0 * normale_bidim(x, z, params_0)
    alpha_1 = weight_1 * normale_bidim(x, z, params_1)
    # l'interprétation de la formule pour Qi en langage d'expressions de NumPy
    return np.array([alpha_0 / (alpha_0 + alpha_1), alpha_1 / (alpha_0 + alpha_1)]).T


# Un pas de l'étape M pour l'algorithme EM
def M_step(data, current_Q, current_params, current_weights):
    # récupération de x et z
    x, z = data[:, 0], data[:, 1]

    # l'interprétation des formules pour des poids en langage d'expressions de NumPy
    new_weights = current_Q.sum(axis=0) / current_Q.sum()

    new_params = np.zeros_like(current_params)

    # l'interprétation des formules pour des paramètres en langage d'expressions de NumPy
    new_params[:, 0] = np.sum(current_Q * x.reshape(-1, 1), axis=0) / np.sum(current_Q, axis=0)
    new_params[:, 1] = np.sum(current_Q * z.reshape(-1, 1), axis=0) / np.sum(current_Q, axis=0)

    x_diff_mean = np.ones((x.shape[0], current_params.shape[0])) * x.reshape(-1, 1) - new_params[:, 0]
    new_params[:, 2] = np.sqrt(
        np.sum(current_Q * x_diff_mean ** 2, axis=0) / np.sum(current_Q, axis=0)
    )
    z_diff_mean = np.ones((z.shape[0], current_params.shape[0])) * z.reshape(-1, 1) - new_params[:, 1]
    new_params[:, 3] = np.sqrt(
        np.sum(current_Q * z_diff_mean ** 2, axis=0) / np.sum(current_Q, axis=0)
    )

    new_params[:, 4] = np.sum(current_Q * x_diff_mean * z_diff_mean / new_params[:, 2] / new_params[:, 3], axis=0) / \
                       np.sum(current_Q, axis=0)

    return new_params, new_weights


def find_bounds(data, params, sigma_mult=2.0):
    # récupération des paramétres
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    # calcul des coins
    x_min = min(mu_x0 - sigma_mult * sigma_x0, mu_x1 - sigma_mult * sigma_x1, data[:, 0].min())
    x_max = max(mu_x0 + sigma_mult * sigma_x0, mu_x1 + sigma_mult * sigma_x1, data[:, 0].max())
    z_min = min(mu_z0 - sigma_mult * sigma_z0, mu_z1 - sigma_mult * sigma_z1, data[:, 1].min())
    z_max = max(mu_z0 + sigma_mult * sigma_z0, mu_z1 + sigma_mult * sigma_z1, data[:, 1].max())

    return x_min, x_max, z_min, z_max


# Réalisation de l'algoritme EM
# 1) pour visualiser l'évolution des paramètres des normales il faut mettre une valeur True à la variable
#    "plot_each_iter" afin que le système dessine des normales à chaque itération
# 2) pour obtenir comme résultat une liste des couples (paramètres, poids) calculées à chaque itération il faut
#    mettre une valeur True à la variable "return_list"
def algorithm_em(data, initial_params, initial_weights, plot_each_iter=False, return_list=False, eps=1e-10,
                 maxiter=100, sigma_mult=2.0):
    # Initialisation des arguments de l'algoritme EM
    res = []
    current_params = np.array(initial_params).copy()
    current_weights = np.array(initial_weights).copy()
    if return_list:
        res.append([current_params.copy(), current_weights.copy()])
    prev_params, prev_weights = np.empty_like(current_params), np.empty_like(current_weights)

    # Si plot_each_iter==True, on dessine des normales pour les paramètres et pour les poids initiaux
    if plot_each_iter:
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        dessine_normales(data, current_params, current_weights, find_bounds(data, current_params, sigma_mult), ax)

    # Le processus itératif
    # On ne va pas faire plus que "maxiter" itérations
    for i in range(maxiter):

        # On copie une couple précédente
        prev_params = current_params.copy()
        prev_weights = current_weights.copy()

        # On calcule Qi courant (l'étape E)
        current_Q = Q_i(data, current_params, current_weights)
        # On calcule la couple courante (l'étape M)
        current_params, current_weights = M_step(data, current_Q, current_params, current_weights)

        # Si plot_each_iter==True, on dessine des normales pour les paramètres et pour les poids courants
        if plot_each_iter:
            fig = plt.figure(i+1)
            ax = fig.add_subplot(111)
            dessine_normales(data, current_params, current_weights, find_bounds(data, current_params, sigma_mult), ax)

        # Si return_list==True, on ajoute la couple courante à la fin de notre liste
        if return_list:
            res.append([current_params.copy(), current_weights.copy()])

        # Si la norme euclidienne entre la couple courante et précédente est inférieure à "eps", ça veut dire que
        # la couple n'a pas été changée significativement à sens de "eps"
        # Donc on ne va pas continuer en ce cas
        if np.linalg.norm(
            np.concatenate((current_params.ravel(), current_weights)) -
            np.concatenate((prev_params.ravel(), prev_weights))
        ) < eps:
            break

    # Si return_list==True, on retourne une liste avec les résultats
    # Sinon, on retourne la dernière couple "courante"
    if return_list:
        return res
    return [current_params.copy(), current_weights.copy()]


# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds(data, res_EM):
    bounds = np.asarray(find_bounds(data, res_EM[0][0]))
    for param in res_EM:
        new_bound = find_bounds(data, param[0])
        for i in [0, 2]:
            bounds[i] = min(bounds[i], new_bound[i])
        for i in [1, 3]:
            bounds[i] = max(bounds[i], new_bound[i])
    return bounds


# la fonction appelée à chaque pas de temps pour créer l'animation
def animate(i):
    ax.cla()
    dessine_normales(data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str(i))
    print("step animate = %d" % i)


if __name__ == '__main__':

    # Le code qui correspond à la question 1
    data = read_file("2015_tme4_faithful.txt")

    # Le code qui correspond à la question 2
    # On vérifie que les valeurs obtenues par normale_bidim sont égales aux valeurs indiquées au fichier de TME
    print(normale_bidim(1, 2, (1.0, 2.0, 3.0, 4.0, 0)) - 0.013262911924324612)
    print(normale_bidim(1, 0, (1.0, 2.0, 1.0, 2.0, 0.7)) - 0.041804799427614503)

    # Le code qui correspond à la question 3
    dessine_1_normale((1.0, 2.0, 1.0, 2.0, 0.3))

    # Le code qui correspond à la question 4
    # affichage des données : calcul des moyennes et variances des 2 colonnes
    mean1 = data[:, 0].mean()
    mean2 = data[:, 1].mean()
    std1 = data[:, 0].std()
    std2 = data[:, 1].std()

    # les paramétres des 2 normales sont autour de ces moyennes
    params = np.array([(mean1 - 0.2, mean2 - 1, std1, std2, 0), (mean1 + 0.2, mean2 + 1, std1, std2, 0)])
    weights = np.array([0.5, 0.5])
    bounds = find_bounds(data, params)

    # affichage de la figure
    fig = plt.figure(-1)
    ax = fig.add_subplot(111)
    dessine_normales(data, params, weights, bounds, ax)

    # Le code qui correspond à la question 5
    # On vérifie que les valeurs obtenues par Q_i sont égales aux valeurs indiquées au fichier de TME
    # current_params = np.array ( [(mu_x, mu_z, sigma_x, sigma_z, rho), # params 1ére loi normale
    #                              (mu_x, mu_z, sigma_x, sigma_z, rho)] ) # params 2éme loi normale
    current_params = np.array([[3.28778309, 69.89705882, 1.13927121, 13.56996002, 0.],
                               [3.68778309, 71.89705882, 1.13927121, 13.56996002,
                                0.]])  # current_weights = np.array ( [ pi_0, pi_1 ] )
    current_weights = np.array([0.5, 0.5])
    T = Q_i(data, current_params, current_weights)
    print(T)

    print('######################################################################')

    current_params = np.array([[3.2194684, 67.83748075, 1.16527301, 13.9245876, 0.9070348],
                               [3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
    current_weights = np.array([0.49896815, 0.50103185])
    T = Q_i(data, current_params, current_weights)
    print(T)

    print('######################################################################')

    # Le code qui correspond à la question 6
    # On vérifie que les valeurs obtenues par M_step sont égales aux valeurs indiquées au fichier de TME
    current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
                            (4.2893485, 79.76680985, 0.52047055, 7.04450242, 0.58358284)])
    current_weights = array([0.45165145, 0.54834855])
    Q = Q_i(data, current_params, current_weights)
    print(M_step(data, Q, current_params, current_weights))

    # Le code qui correspond aux questions 7 et 8
    # Les tâches des questions 7 et 8 ont été résolues en implémentant une seule fonction "algorithm_em"
    # Pour la question 7, il faut effectuer plot_each_iter = True, return_list = False
    # Pour la question 8, il faut effectuer plot_each_iter = False, return_list = True
    # Plus d'info aux commentaires qui correspondent à cette fonction

    figi = 0

    # Pour obtenir les résultat correspondants à la question 7, il faut décommenter le code ci-dessous
    # algorithm_em(data, params, weights, plot_each_iter=True, return_list=False, maxiter=4)
    # figi = 5

    # On sauvegarde une liste de tous les couples à chaque l'itération
    res_EM = algorithm_em(data, params, weights, plot_each_iter=False, return_list=True, maxiter=20, eps=1e-4)

    # Le code pour calculer les bornes sans l'utilisation de variables globales
    bounds = find_video_bounds(data, res_EM)
    count = 0
    print('######################################################################')
    for it_res in res_EM:
        print('Itération {0}\nParamètres\n{1}\nPoids\n{2}\n\n'.format(count, it_res[0], it_res[1]))
        count += 1

    # création de l'animation : tout d'abord on crée la figure qui sera animée
    fig = plt.figure(figi)
    ax = fig.add_subplot(111)
    plt.axis(bounds)

    # exécution de l'animation
    anim = animation.FuncAnimation(fig, animate, frames=len(res_EM), interval=10)

    plt.show()

    # éventuellement, sauver l'animation dans une vidéo
    # anim.save('old_faithful.avi', bitrate=4000)
