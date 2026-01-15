import pytest
import numpy as np
from maze_genetique import GeneticSolver

#On lancera ce fichier avec la commande pytest test_genetique.py


@pytest.fixture
def solver():
    matrice_vide = np.zeros((10, 10), dtype=int)
    return GeneticSolver(matrice_vide, (0, 0), (9, 9))

def test_initialisation_directions(solver):
    """Vérifie que le dictionnaire des directions est conforme"""
    assert 0 in solver.directions
    assert solver.directions[0] == (0, 1)
    assert solver.directions[7] == (1, 1)

def test_genese_taille(solver):
    """Vérifie que la matrice de population a la bonne taille"""
    pop_size = 50
    genome_len = 30
    population = solver.genese(pop_size, genome_len)
    
    assert population.shape == (pop_size, genome_len) 
    assert len(solver.get_population()) == pop_size

def test_genese_valeurs(solver):
    """Vérifie que les gènes sont bien des directions valides [0, 7]"""
    population = solver.genese(100, 50)
    assert np.all(population >= 0)
    assert np.all(population <= 7)

def test_endCell(solver):
    """Vérifie que enCell renvoie la nouvelle position et le nombre de steps"""
    #on modif la matrice pour le test
    solver.matrix[0, 0] = 1
    solver.matrix[0, 1] = 1

    genome = [0] 

    pos, steps = solver.endCell(genome)
    
    # On atterri en (0, 1) car on à ouvert la matrice et 1 pas car la taille du génome est de 1
    assert pos == (0, 1)
    assert steps == 1

def test_fitness(solver):
    """Vérifie que fitness renvoie le bon score de pénalité"""
    genome = [0, 0, 0] 
    L = 100 #taille génome
    
    score = solver.fitness(genome, L)
    
    # Vérif du calcul :
    # Distance Manhattan (0,0) -> (9,9) = |0-9| + |0-9| = 18
    # Steps = 0 (bloqué direct)
    # Pénalités : 
    #    steps < L/2 (50)-> + L/10 (10)
    #    steps < L/5 (20)-> + L/5 (20)
    #    steps == 0-> + L (100)
    # Total = 18 + 10 + 20 + 100 = 148
    
    assert score == 148

def test_selection(solver):
    """Vérifie que selection sékectionne bien ts * population"""
    # population de 10 individus de taille 5
    solver.population = np.zeros((10, 5), dtype=int)
    
    ts = 0.4
    L = 5
    best_score = solver.selection(ts, L)
    
    assert len(solver.population) == 4
    assert isinstance(best_score, (float, np.floating, int))

def test_reproduction(solver):
    """Vérifie que l'enfant est un mélange et que la fonction atteint bien la cuble de taille de pop """
    solver.population = np.zeros((5, 10), dtype=int) 
    
    target_size = 20
    genome_len = 10
    
    nb_enfants = solver.reproduction(target_size, genome_len)
    
    assert nb_enfants == 15 # 20 - 5 
    assert len(solver.population) == 20 # ça finit à 20
    assert solver.population.shape == (20, 10)

    # père : ue des 1
    # mère : que des 2
    p1 = np.ones((1, 10), dtype=int) * 1
    p2 = np.ones((1, 10), dtype=int) * 2
    solver.population = np.vstack((p1, p2))
    
    solver.reproduction(3, 10)
    
    child = solver.population[2]
    
    # L'enfant doit contenir des un et des 2
    assert 1 in child
    assert 2 in child

def test_mutation(solver):
    """Vérifie que la mutation modifie bien la population"""
    np.random.seed(42)
    
    solver.population = np.zeros((10, 10), dtype=int) # -> on ne met que des Zéros pour se facilliter la tâche

    solver.mutation(tm=10.0) 
    
    assert np.sum(solver.population) > 0 # maintenant qu'il n'y à plus que des zéros
