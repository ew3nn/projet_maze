import time
import numpy as np
import matplotlib.pyplot as plt
from maze_generator import Labyrinthe
from maze_solver import MazeResolution
from maze_genetique import GeneticSolver


def main():
    TAILLE = 50
    print(f"--- Génération du Labyrinthe {TAILLE}x{TAILLE} ---")

    start_x = 0
    start_y = 0
    #Génération
    laby = Labyrinthe(TAILLE)
    start = time.time()
    laby.dfs_iteratif(start_x, start_y)
    print(f"Génération terminée en {time.time() - start:.4f} sec")
    
    # Sauvegarde visuelle du labyrinthe brut
    plt.imsave("labyrinthe_brut.png", laby.matrix, cmap="binary_r")

    # Résolution Dijkstra Optimisé (Itératif)
    print("\n--- Dijkstra Optimisé (BFS) ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    start = time.time()
    map_opti = solver_opti.dijkstra_opti()
    print(f"Calcul terminé en {time.time() - start:.4f} sec")
    solver_opti.afficher_carte_couts("dijkstra_iteratif.png")

    # 3. Résolution Dijkstra (avec le même Goal pour comparer)
    print("\n--- Dijkstra Itératif ---")
    # On passe le goal du premier solver pour être sûr de viser le même point
    solver_rec = MazeResolution(laby.get_matrix().copy(), goal=solver_opti.goal)
    start = time.time()
    map_rec = solver_rec.dijkstra()
    print(f"Calcul terminé en {time.time() - start:.4f} sec")
    solver_rec.afficher_carte_couts("dijkstra.png")

    # Vérification que les deux méthodes donnent le même résultat
    difference = np.sum(map_opti != map_rec)
    if difference == 0:
        print("\nSuccès : Les deux méthodes (Itérative et Récursive) donnent le même résultat")
    else:
        print(f"\n Erreur : {difference} différences trouvées entre les cartes")

    laby = Labyrinthe(8)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 8 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    laby = Labyrinthe(16)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 16 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    laby = Labyrinthe(32)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 32 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    laby = Labyrinthe(64)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 64 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    laby = Labyrinthe(128)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 128 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    laby = Labyrinthe(256)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 256 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    laby = Labyrinthe(512)
    laby.dfs_iteratif(start_x, start_y)
    print("\n--- Dijkstra sur labyrinthe taille 512 ---")
    solver_opti = MazeResolution(laby.get_matrix().copy())
    map_opti = solver_opti.dijkstra_opti()
    solver_opti.chemin()

    print("\n --- On passe au sous projet 3 avec les algos génétiques---")

    #on recréer un labyrinthe pour notre experience
    N = 10
    laby = Labyrinthe(N)
    laby.dfs_iteratif(start_x, start_y)
    M= laby.get_matrix()
    #on utilise djikstra pour trouver le point le plus éloigné de notre point de départ comme demandé
    solver = MazeResolution(M.copy())
    map = solver.dijkstra_opti()

    max_idx = np.argmax(map.astype(float))#ici on récupère la valeur la plus éloignée de l'objectif, donc notre futur start
    start = np.unravel_index(max_idx, map.shape)#on se sert ensuite de cette valeur pour la convertir en l'index de la matrice, cette fonction permet de donner l'indice la ou la valeur trouvée est
    goal = solver.goal #on récupère l'objectif

    print(f"Départ fixé en {start} (Distance: {map[start]})")
    print(f"Objectif fixé en {goal}")

    print("\n --- GENESE---")
    POPULATION_SIZE = 100
    GENOME_LENGTH = N * 3

    gen_solver = GeneticSolver(M.copy(), start, goal)
    population = gen_solver.genese(POPULATION_SIZE, GENOME_LENGTH)

    # Juste pour voir à quoi ressemble le premier individu
    print("\nExemple du premier individu (liste de directions) :")
    print(population[0])
    
    print("\n --- Evolution---")

    TAUX_SELECTION = 0.3
    best_score = gen_solver.selection(TAUX_SELECTION, GENOME_LENGTH)

    print(f"Meilleur score : {best_score}")
    print(f"Nouvelle taille de la population : {len(gen_solver.get_population())}")

    best = gen_solver.get_population()[0]
    print(f"Génome du meilleur (taille {len(best)}) : {best}")

    pos_finale, steps = gen_solver.endCell(best)
    print(f"Le best s'est arrêté en {pos_finale} après {steps} pas.")
    print(f"Distance au but : {abs(pos_finale[0] - goal[0]) + abs(pos_finale[1] - goal[1])}")


    print("\n--- Lancement de l'Algorithme Génétique Complet ---")
    
    # HYPERPARAMETR§ES
    MAX_GEN = 10
    TS = 0.5
    TM = 0.08
    POPULATION_SIZE = 5
    GENOME_LENGTH = N * 3
    USE_PHEROMONES = True
    
    # Nouvelle génèse
    print("Réinitialisation de la population...")
    gen_solver.genese(POPULATION_SIZE, GENOME_LENGTH)
    
    best_solution, history = gen_solver.solve(
        nG=MAX_GEN, 
        ts=TS, 
        tm=TM, 
        target_pop_size=POPULATION_SIZE, 
        genome_length=GENOME_LENGTH,
        use_pheromones=USE_PHEROMONES
    )

    # loss
    plt.figure(figsize=(10, 5))
    plt.plot(history, label="Fitness (Loss)")
    plt.xlabel("Générations")
    plt.ylabel("Meilleur Score (Distance + Pénalités)")
    plt.title(f"Convergence AG (Pheromones={USE_PHEROMONES})")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence_genetique.png")
    print("Graphique de convergence sauvegardé : convergence_genetique.png")

    # résultat sur le labyrinthe 
    plt.figure(figsize=(8, 8))
    # matrice finale
    plt.imshow(gen_solver.matrix, cmap="binary_r") 
    
    # chemin du champion (même logique que le traçage du chemin rouge)
    if best_solution is not None:
        path = []
        curr = start
        path.append(curr)

        original_matrix = M

        for gene in best_solution:
            di, dj = gen_solver.directions[gene]
            ni, nj = curr[0] + di, curr[1] + dj
            
            # On vérifie si on peut avancer dans la mtrice modif
            if 0 <= ni < N and 0 <= nj < N and original_matrix[ni, nj] != 0:
                curr = (ni, nj)
                path.append(curr)
                if curr == goal: break
            else:
                break
        
        py, px = zip(*path)
        plt.plot(px, py, color='red', linewidth=2, label='Solution AG')
        print(f"Longueur du chemin trouvé : {len(path)}")

    plt.plot(start[1], start[0], 'go', label='Départ')
    plt.plot(goal[1], goal[0], 'r*', markersize=15, label='But')
    
    plt.legend()
    plt.title(f"Résultat Final (Gen {len(history)})")
    plt.axis('off')
    plt.savefig("resultat_final_genetique.png")
    print("Image du résultat final sauvegardée : resultat_final_genetique.png")
    plt.show()

if __name__ == "__main__":
    main()
