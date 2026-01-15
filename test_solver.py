import pytest
import numpy as np
from maze_solver import MazeResolution

#On lancera ce fichier avec la commande pytest test_solver.py

@pytest.fixture
def resolution_test():
    """
    Crée un petit labyrinthe 3x3 simple pour vérifier les calculs manuellement
    """
    matrice = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 1]
    ])
    # goal en 2,2 pour pouvoir reproduire les tests
    return MazeResolution(matrice, goal=(2, 2))


def test_initialisation_map(resolution_test):
    """Vérifie que la carte est bien initialisée correctement"""
    solver = resolution_test
    
    # mur
    assert solver.map[0, 2] == -1
    
    # chemin
    assert solver.map[0, 0] is None
    
    # taille
    assert solver.map.shape == (3, 3)

def test_get_adjacents(resolution_test):
    """Vérifie que la fonction renvoie bien les voisins valides"""
    solver = resolution_test
    
    # Test coin haut gauche (0,0) -> 3 voisins
    voisins_00 = solver.get_adjacents(0, 0)
    attendu = [(0, 1), (1, 0), (1, 1)]
    
    assert len(voisins_00) == 3
    assert sorted(voisins_00) == sorted(attendu)

    # Test centre -> 8 voisins
    voisins_11 = solver.get_adjacents(1, 1)
    assert len(voisins_11) == 8

def test_goal_aleatoire(resolution_test):
    """Vérifie que le goal choisit aléatoirement est valide"""
    solver = resolution_test
    solver.goal = None
    
    solver.def_goal_aleatoire()
    
    gx, gy = solver.goal
    assert solver.map[gx, gy] != -1
    assert 0 <= gx < 3 and 0 <= gy < 3

def test_dijkstra_opti(resolution_test):
    """Vérifie que le BFS calcule les bonnes distances"""
    solver = resolution_test
    map = solver.dijkstra_opti()
    # Le But est à 0
    assert map[2, 2] == 0
    # Le voisin direct à 1
    assert map[2, 1] == 1
    # voisin diagonal à 1
    assert map[1, 1] == 1
    # distance de 2 avec la distance de manhattan
    assert map[0, 0] == 2

def test_dijkstra_(resolution_test):
    """Vérifie que djikstra calcul bien les bonnes distances"""
    solver = resolution_test
    map = solver.dijkstra()
    # Le But est à 0
    assert map[2, 2] == 0
    # Le voisin direct à 1
    assert map[2, 1] == 1
    # voisin diagonal à 1
    assert map[1, 1] == 1
    # distance de 2 avec la distance de manhattan
    assert map[0, 0] == 2

def test_chemin(resolution_test):
    """Vérifie que la fonction retrouve le bon chemin"""
    solver = resolution_test
    solver.dijkstra_opti() #connaitre la map avant
    path = solver.chemin(start=(0, 2))
    assert path == []
    path = solver.chemin(start=(0, 0))
    
    # Vérification de si on part du bon endroit et si on finit au bon endroit
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) > 0
    
    # étape suivant = étape ancienne + 1
    for k in range(len(path) - 1):
        curr = path[k]
        next_step = path[k+1]
        
        # Calcul de distance 
        dist_x = abs(curr[0] - next_step[0])
        dist_y = abs(curr[1] - next_step[1])
        assert dist_x <= 1 and dist_y <= 1

