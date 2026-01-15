import numpy as np
import random
import matplotlib.pyplot as plt


class Labyrinthe:

    def __init__(self, taille):
        self.taille = taille
        self.matrix = np.zeros((taille, taille), dtype=int)

    def voisin_relie(self, x, y):
        """
        Evalue les voisins d'une celulle pour savoir si elles sont reliées à un chemin ou un mur
        param self: class Labyrinthe
        param x: position sur l'axe x de la cellule
        param y: position sur l'axe y de la cellule

        """
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
        """
        Evalue les voisins d'une cellule pour savoir si ils sont valides ou non 
        param self: class Labyrinthe
        param x: position sur l'axe x de la cellule
        param y: position sur l'axe y de la cellule

        """
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
        """
        Algo de DFS pour remplir notre matrice de manière itérative 
        param self: class Labyrinthe
        param x: position sur l'axe x de la cellule
        param y: position sur l'axe y de la cellule

        """
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
    
    def get_matrix(self):
        """
        Fonction pour récupérer la matrice dans les autres dossiers
        """
        return self.matrix
