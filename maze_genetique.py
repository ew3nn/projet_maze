import numpy as np
import random

class GeneticSolver :
    def __init__(self, matrix, start, goal):
        """
        matrix : la matrice du labyrintht
        start : point de départ (x, y)
        goal : arrivée (x, y)
        """
        self.matrix = matrix
        self.N = matrix.shape[0]
        self.start = start
        self.goal = goal
        self.population = []

        #On créer un dictionnaire avec les chiffres en indices qui correspondent à une clé qui est le déplacement
        self.directions = {
            0: (0, 1),
            1: (-1, 1),
            2: (-1, 0),
            3: (-1, -1),
            4: (0, -1),
            5: (1, -1),
            6: (1, 0),
            7: (1, 1)
        }

    def genese(self, population_size, genome_length):
        """
        Génère la population initiale
        population_size P :nb individus
        genome_length L : nb de mvmts par individu
        """
        #matrice de nos programmes, suite de chiffre comprise entre 0 et 7 inclus de taille L colonnes et P lignes
        self.population = np.random.randint(0, 8, size=(population_size, genome_length))
        
        print(f"Génèse terminée : {population_size} individus créés avec {genome_length} gènes chacun.")
        return self.population

    def get_population(self):
        return self.population
    
    def endCell(self, C):
        """
        Execute tout les steps d'un individus créer
        retourne la position de la cellule atteinte à l’extrémité du plus long sous-chemin
        qui relie de manière valide cette extrémité au point de départ de C.
        param C : programme/chemin genome
        """
        cur_i, cur_j = self.start
        steps = 0
        for gene in C :
            direction = gene
            di, dj = self.directions[direction]

            next_i, next_j = cur_i + di, cur_j + dj #on retrace vraiment le chemin à chaque fois pour chaque genome
            if (0 <= next_i < self.N and 0 <= next_j < self.N and self.matrix[next_i, next_j] != 0):
                cur_i, cur_j = next_i, next_j
                steps += 1
                
                #si on atteint le but on arrete tout et on à gagner on à créer le génome parfait
                if (cur_i, cur_j) == self.goal:
                    break
            else:#pas réussi ou alorson s'est pris un mur
                break
                
        return (cur_i, cur_j), steps

    def fitness(self, genome, genome_lenght):
        """
        Notre fonction de rewarding
        Plus elle est basse mieux c'est
        """
        penalties = 0
        curr_i, curr_j = self.start
        (end_i, end_j), steps = self.endCell(genome)
        (goal_i, goal_j) = self.goal

        visited = set()
        visited.add((curr_i, curr_j))
        
        steps = 0
        backtracks = 0
        
        # On simule le chemin manuellement ici pour avoir le contrôle sur le backtracs qui est embetant
        for gene in genome:
            di, dj = self.directions[gene]
            next_i, next_j = curr_i + di, curr_j + dj
            
            # Vérification Mur / Hors Map
            if not (0 <= next_i < self.N and 0 <= next_j < self.N and self.matrix[next_i, next_j] != 0):
                # On s'arrête net si on tape un mur
                break
            
            # Vérification Backtracking
            if (next_i, next_j) in visited:
                backtracks += 1
            else:
                visited.add((next_i, next_j))
            
            curr_i, curr_j = next_i, next_j
            steps += 1
            
            if (curr_i, curr_j) == self.goal:
                break
        
        #pour savoir notre distance au but, on utilise la distance de manhattan
        #  c'est ce qu'il y à de plus précis sur une grille
        distance = abs(end_i - goal_i) + abs(end_j - goal_j)

        penalties += backtracks
        if distance == 0:
            penalties -= 500
        else :
            penalties += 20

        murs_autour = 0
        card =[
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
                (1, 0),
                (1, 1)
            ]
        for dx, dy in card:
                nx, ny = end_i + dx, end_j + dy
                if not (0 <= nx < self.N and 0 <= ny < self.N) or self.matrix[nx, ny] == 0:
                    murs_autour += 1
    
        if steps == 0 :
            penalties += genome_lenght #cette solution est forcément mauvaise donc on veut la virer
        if murs_autour >= 7:
            penalties += 5
        penalties -= steps*0.2

        return penalties
    
    def selection(self, ts, genome_lenght):
        """
        Evalue les génomes afin de ne garder que les meilleurs en fonctions du ts
        ts : taux de sélection comprix entre 0 et 1
        """
        scores = []
        for genome in self.population:
            score = self.fitness(genome, genome_lenght)
            scores.append(score)
        
        scores = np.array(scores)
        
        sorted_scores = np.argsort(scores) #-> ordre croissant des indices de score en fonction de leur valeurs car on veut peu de penalties ex [100, 0] -> [1, 0]
        
        n_best = int(len(self.population) * ts)
        best_indices = sorted_scores[:n_best]
        self.population = self.population[best_indices] #-> on ne garde que le pourcentage voulu de meilleurs
        best = scores[sorted_scores[0]]#-> afin de suivre si la population évolue bien et si on se rapproche du but
        return best
    
    def reproduction(self, target_pop_size, genome_length):
        """
        Reproduction (cross-over)
        On complète la population avec des enfants jusqu'à atteindre target_pop_size.
        target_pop_size = taille de la population souhaité
        genome_lenght : taille des génomes
        """
        children = []
        nb_children = target_pop_size - len(self.population)
        
        survivors = self.population 
        
        for i in range(nb_children):
            idx1 = np.random.randint(0, len(survivors))
            idx2 = np.random.randint(0, len(survivors))
            
            pere = survivors[idx1]
            mere = survivors[idx2]
            
            # Préselection de notre plage de cut pour ne pas que ce soit vraiment du hasard
            min_cut = int(genome_length * 0.3)
            max_cut = int(genome_length * 0.7)
            cut = np.random.randint(min_cut, max_cut) 
            
            child = np.concatenate((pere[:cut], mere[cut:])) # -> création de l'enfant
            children.append(child)
            
        if children:
            children_array = np.array(children)
            self.population = np.vstack((self.population, children_array))
            
        return len(children) # pour les test unitaires afin de s'assurer qu'il y à bien le bon nombre d'enfant
    
    def mutation(self, tm):
        """
        Modifie aléatoirement certains genomes
        tm : taux de mutation [0, 1] -> si on veut quelques chose qui finit par arriver à la solution on doit le garder assez bas
        """
        
        nb_mutations = int(self.population.size * tm)        
        for i in range(nb_mutations):
            idx_individu = np.random.randint(0, len(self.population))
            
            idx_gene = np.random.randint(0, self.population.shape[1])
            
            random_gene = np.random.randint(0, 8)
            self.population[idx_individu, idx_gene] = random_gene

    def pheromones(self):
        """
        Si une case explorée par un génome est entouré de 7 murs donc un cul de sac, on la ferme
        """
        # stop de la génération actuelle
        stops = []
        for genome in self.population:
            pos, _ = self.endCell(genome)
            stops.append(pos)
        
        stops = set(stops)

        SAFE_RADIUS = 5
        
        for (x, y) in stops:
            if (x, y) == self.goal or (x, y) == self.start: # -> ligne de sécurité car le goal peut être un cul de sac
                continue

            dist_to_start = abs(x - self.start[0]) + abs(y - self.start[1]) #-> ajout de cette ligne car on a eu des problèmes dans lesquelles les phéromones bouchainet entièrement le passage
            if dist_to_start <= SAFE_RADIUS:
                continue
            
            murs_autour = 0
            card =[
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
                (1, 0),
                (1, 1)
            ]
            
            for dx, dy in card:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.N and 0 <= ny < self.N) or self.matrix[nx, ny] == 0:
                    murs_autour += 1
            # condamnation
            if murs_autour >= 7:
                self.matrix[x, y] = 0 

    def solve(self, nG, ts, tm, target_pop_size, genome_length, use_pheromones):
        """
        Boucle d'évolution complète
        Paramètres:
        nG : nombre de générations max
        ts : taux de sélection 
        tm : taux de mutation 
        target_pop_size : taille de la population stable
        """
        loss = []
        best_solution_found = None
        best_genome_ever = None
        best_fitness_ever = float('inf')
        
        print(f"Début de l'évolution sur {nG} générations...")
        
        for g in range(nG):
            # fitness et sélectio,
            # best score
            best_score = self.selection(ts, genome_length)
            loss.append(best_score)
            
            # Récupération du meilleur individu pour vérifier si on a gagné
            champion = self.population[0]
            print(f"voici le champion {champion}")
            print(f"voici la population {self.population}")

            if best_score < best_fitness_ever: #afin de tracer la courbe sur le graphique du labyrinthe même si la solution n'est pas trouvée
                best_fitness_ever = best_score
                best_genome_ever = champion.copy() 

            
            (pos, steps) = self.endCell(champion)
            
            # solution trouvé
            if pos == self.goal:
                print(f"Réussite !! au bout de {g} génération")
                best_solution_found = champion
                break
                
            if use_pheromones:
                self.pheromones()
   
            # reproduction
            self.reproduction(target_pop_size, genome_length)
            
            # mutation
            self.mutation(tm)
            
            # suivi de l'avancée
            if g % 10 == 0:
                print(f"Gen {g}: Best Fitness = {best_score:.2f} - Pos: {pos}")

        if best_solution_found is not None:
            print("tentative réussite")
            return best_solution_found, loss
        else:
            print("meilleur tentative")
            return best_genome_ever, loss


