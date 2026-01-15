import numpy as np
import random
import matplotlib.pyplot as plt
import time

class labyrinthe:

    def __init__(self, taille):
        self.taille = taille
        self.matrix = np.zeros((taille, taille), dtype=int)

    def voisin_relie(self, x, y):
        directions = [
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)
        ]
        bon = 0
        for dx, dy in directions :
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.taille and 0 <= ny < self.taille:
                if self.matrix[nx][ny] == 1:
                    bon += 1
        return bon == 1

    def voisins_valides(self, x, y):
        directions = [
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1)
        ]
        voisins = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.taille and 0 <= ny < self.taille:
                if self.matrix[nx, ny] == 0 and self.voisin_relie(nx, ny):
                    voisins.append((nx, ny))
        return voisins

    def dfs_iteratif(self, x, y):
        stack = [(x, y)]
        self.matrix[x][y] = 1

        while stack:
            px, py = stack[-1] #previous coordonnées stack[-1] = dernier élément de la pile

            voisins = self.voisins_valides(px, py)

            if voisins:
                nx, ny = random.choice(voisins)#fctn pour sélectionner un aléatoire entre tous les voisins présents da,s la liste
                self.matrix[nx][ny] = 1
                stack.append((nx, ny))
            else:
                stack.pop()

    def afficher_matplotlib(self):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.matrix, cmap="binary_r") # noir = mur, blanc = passage 
        plt.axis("off")
        plt.show()


class MazeResolution :

    def __init__(self, matrice, goal = None):
        self.M = matrice
        self.N = matrice.shape[0]
        self.goal = goal
        self.map = self.initialize_map()
        self.direction_map = None

    def initialize_map(self):
        map = np.full((self.N, self.N), None, dtype=object)
        map[self.M == 0 ] = -1 #ligne pour associer les murs à la valeur -1
        return map

    def get_adjacents(self, i, j):
        adjacents = []
        pos_possible = [-1, 0, 1] #position que l'on peut prendre c'est le carré autour de la cellule 
        for di in pos_possible:
            for dj in pos_possible :

                if di == 0 and dj == 0 : #ça c'est la cellule avec elle même donc ça ne compte pas
                    continue
                k = i + di
                l = j + dj

                if 0 <= k < self.N and 0 <= l < self.N: #voisin valide et qui ne sort pas de la matrice
                    adjacents.append((k, l))
        return adjacents
    
    def dijkstra(self):
        
        i_g, j_g = None, None #c'est ici qu'on définit notre goal, par question de simplicité
        while i_g is None: 
            i = random.randint(0, self.N - 1)
            j = random.randint(0, self.N - 1)
            if self.map[i, j] != -1:
                i_g, j_g = i, j #goal trouvé donc on sort de la boucle
                self.goal = (i_g, j_g)
        
        counter = 0
        self.map[i_g, j_g] = 0

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
    
    def dijkstras_opti(self):
        i_g, j_g = None, None #c'est ici qu'on définit notre goal, par question de simplicité
        while i_g is None: 
            i = random.randint(0, self.N - 1)
            j = random.randint(0, self.N - 1)
            if self.map[i, j] != -1:
                i_g, j_g = i, j #goal trouvé donc on sort de la boucle
                self.goal = (i_g, j_g)
        
        counter = 0
        self.map[i_g, j_g] = 0

        queue = [self.goal] #on part du goal et on le met en premier dans notre liste son cout est 0

        while queue :
            current_i, current_j = queue.pop(0) #on récupère la position d'ou on part
            current_cost = self.map[current_i, current_j]

            for ni, nj in self.get_adjacents(current_i, current_j): #on s'occupe encore des voisins avec otre fonction
                if self.map[ni, nj] is None: 
                    
                    self.map[ni, nj] = current_cost + 1
                    queue.append((ni, nj))

        self.map[self.map == None] = -1
        
        return self.map.copy()
    
    def afficher_carte(self):
        """Affiche la carte de coûts du Dijkstra."""
        N = self.N
        
        matrix_float = np.array(self.M, dtype=np.float32)
        matrix_float[self.M == -1] = np.nan 
        
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix_float, cmap="viridis", origin='upper') 
        
        cbar = plt.colorbar(label='Coût (Distance au But)')
        cbar.set_ticks(np.linspace(np.nanmin(matrix_float), np.nanmax(matrix_float), 5, dtype=int))
        
        # Affichage du But
        plt.plot(self.goal[1], self.goal[0], 'r*', markersize=15, label='But') # (col, row) pour l'affichage
        plt.text(self.goal[1] + 0.5, self.goal[0], 'But', color='red', fontsize=12, fontweight='bold')
        
        plt.title('map coloré en fonction du cout', fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.grid(True, color='k', linestyle='-', linewidth=0.1)
        plt.tight_layout()
        plt.savefig("dijkstra_map.png")


laby = labyrinthe(50)
M = laby.matrix
start_x = 0
start_y = 0
start = time.time()
laby.dfs_iteratif(start_x, start_y)
end = time.time()
print(laby.matrix)
laby.afficher_matplotlib()
print(f"Temps d'exécution : {end - start} secondes")
#complexité = O(1) car on parcours nos 8 voisiins à chaque fois -> indépendant de la taille de la matrice, multiplié par O(n²) -> dépend de la taille
#de la matrice on itère sur toute les cellules donc :
#O(1)* O(n²) = O(n²)

solver_literal = MazeResolution(M.copy()) 
start_time_1 = time.time()
map_literal = solver_literal.dijkstra()
end_time_1 = time.time()
time_literal = end_time_1 - start_time_1

solver_opti = MazeResolution(M.copy(), goal=solver_literal.goal) # Utilise le même But
start_time_2 = time.time()
map_opti = solver_opti.dijkstras_opti()
end_time_2 = time.time()
time_opti = end_time_2 - start_time_2

print("=========================================================")
print(f"| RÉSULTATS DES TESTS DIJKSTRA ({'50'}x{'50'}) |")
print("=========================================================")
print(f"But choisi (i, j) : {solver_opti.goal}")
print("\n--- 1. Algorithme Littéral (Scan/O(N^4)) ---")
print(f"Temps d'exécution : {time_literal:.6f} secondes")
print(f"Coût Max atteint : {np.nanmax(map_literal[map_literal != -1])}")

print("\n--- 2. Algorithme Optimisé (BFS/O(N^2)) ---")
print(f"Temps d'exécution : {time_opti:.6f} secondes")
print(f"Coût Max atteint : {np.nanmax(map_opti[map_opti != -1])}")

print("\n--- Comparaison ---")
# Le temps d'exécution est souvent très faible pour N=50. La différence est plus claire pour N=500.
print(f"L'optimisé est {time_literal/time_opti:.2f} fois plus rapide.")

# Affichage de la carte optimisée pour la vérification
solver_literal.afficher_carte()
