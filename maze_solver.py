import numpy as np
import random
import matplotlib.pyplot as plt

class MazeResolution:
    def __init__(self, matrice, goal=None):
        self.M = matrice
        self.N = matrice.shape[0]
        self.goal = goal
        self.map = self.initialize_map()
    
    def initialize_map(self):
        """
        Préparation de l'environnement de djikstra
        """
        map_grid = np.full((self.N, self.N), None, dtype=object)
        map_grid[self.M == 0] = -1 # 0 = mur
        return map_grid

    def get_adjacents(self, i, j):
        adjacents = []
        # Voisinage à 8 cases de manière plus propre que dans le maze generator
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0: continue
                k, l = i + di, j + dj
                if 0 <= k < self.N and 0 <= l < self.N:
                    adjacents.append((k, l))
        return adjacents

    def def_goal_aleatoire(self):
        if self.goal is None:
            while True:
                i = random.randint(0, self.N - 1)
                j = random.randint(0, self.N - 1)
                if self.map[i, j] != -1:
                    self.goal = (i, j)
                    break
        return self.goal

    def dijkstra_opti(self):
        """BFS itératif"""
        self.def_goal_aleatoire()
        # Reset map pour s'assurer qu'on part de propre
        self.map = self.initialize_map()
        self.map[self.goal] = 0
        
        queue = [self.goal]

        while queue:
            ci, cj = queue.pop(0)
            current_cost = self.map[ci, cj]
            
            for ni, nj in self.get_adjacents(ci, cj):
                if self.map[ni, nj] is None:
                    self.map[ni, nj] = current_cost + 1
                    queue.append((ni, nj))
        
        self.map[self.map == None] = -1
        return self.map.copy()
    
    def dijkstra(self):
        self.def_goal_aleatoire()
        if self.goal:
            self.map[self.goal] = 0
        counter = 0
        while np.any(self.map == None):#etape9
            found_new_location = False
            for k in range(self.N):
                for l in range(self.N):
                    
                    if self.map[k, l] == counter:#etape 5
                        
                        for nk, nl in self.get_adjacents(k, l):#etape 6, on recherche nos cases ou l'on peut aller
                            
                            if self.map[nk, nl] is None:
                                self.map[nk, nl] = counter + 1
                                found_new_location = True 

            if found_new_location:
                counter += 1#etape 8
            else:
                break #goal innategniable on est coincé
        self.map[self.map == None] = -1 
        
        return self.map.copy() #toujours renvoyer une copie pour ne pas gêner la map et éviter les effets 
    #désagréables par la suite dans le code

    def chemin(self, start= (0,0)):
        chemin = []
        current = start 
        if self.map[current] == -1 or self.map[current] is None:# si le départ est en dehors de la matrice
            return chemin
        chemin.append(current)
        while current != self.goal:
            ci, cj = current
            current_val = self.map[ci, cj]
            
            found = False
            for ni, nj in self.get_adjacents(ci, cj):
                if self.map[ni, nj] == current_val - 1:
                    current = (ni, nj)
                    chemin.append(current)
                    found = True
                    break
            
            if not found:
                break
        print(f"Longueur du chemin : {len(chemin)}")
        return chemin

    def afficher_carte_couts(self, filename="resultat_dijkstra.png"):
        chemin = self.chemin()
        matrix_float = np.array(self.map, dtype=float)
        # Gestion des murs pour l'affichage (-1 -> NaN)
        matrix_float[self.map == -1] = np.nan
        
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix_float, cmap="viridis")
        plt.colorbar(label="Distance au but")
        
        if self.goal:
            plt.plot(self.goal[1], self.goal[0], 'r*', markersize=15, label='Goal')
        if chemin :
            py, px = zip(*chemin) #zip pour séparer nos coordonnées x et y
            plt.plot(px, py, color='red', linewidth=3, label='Solution')
            
        plt.title(f"Carte des distances (N={self.N})")
        plt.axis('off')
        plt.savefig(filename)
        plt.close() # Ferme la figure pour libérer la mémoire
