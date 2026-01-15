import pytest
import numpy as np
from maze_generator import Labyrinthe

#On lancera ce fichier avec la commande pytest test_generator.py


@pytest.fixture # -> fournit un environnement stable et identique pour chaque test
def laby():
    """Crée un labyrinthe de taille 10*1O pour chaque test"""
    return Labyrinthe(10)

def test_initialisation(laby):
    """Vérifie que la matrice est conforme"""
    assert laby.matrix.shape == (10, 10)
    assert np.all(laby.matrix == 0)
    assert laby.taille == 10

def test_voisin_relie(laby):
    """
    Vérifie la règle de la génération une case n'est validement reliée que si elle touche 
     exactement 1 case déjà visitée
    """
    # aucun voisin visité
    assert laby.voisin_relie(5, 5) is False

    # 1 voisin visité
    laby.matrix[4, 5] = 1 
    assert laby.voisin_relie(5, 5) is True

    # plusieurs voisins visités
    laby.matrix[5, 4] = 1
    assert laby.voisin_relie(5, 5) is False

def test_voisins_valides_selection(laby):
    """Vérifie que la fonction ne retourne que les voisins éligibles"""
    # On est en (5,5)
    # Le voisin (4,5) est un Mur mais il touche un autre chemin en (3,5) -> Validité dépend du contexte
    # on part d'une grille vierge
    
    # On met un chemin en (4, 4)
    laby.matrix[4, 4] = 1
    
    # On demande les voisins valides pour (4, 4)
    # Les voisins possibles sont tout autour
    # Prenons (4, 5), il est à 0, il touche (4,4), il ne touche rien d'autre
    # Donc (4, 5) doit être dans la liste
    
    voisins = laby.voisins_valides(4, 4)
    assert (4, 5) in voisins
    assert (3, 4) in voisins

def test_dfs(laby):
    """Vérifie que l'algo rempli la matrice"""
    laby.dfs_iteratif(0, 0)
    assert laby.matrix[0, 0] == 1
    assert np.sum(laby.matrix) > 1
    assert np.all(np.isin(laby.matrix, [0, 1])) # seulement 0 ou 1 sont dans la matrice
