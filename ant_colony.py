import numpy as np


class AntColony():
    def __init__(self, distances, num_ants, iteration, alpha, beta, evaporation):
        self.distances = distances
        self.n_ants = num_ants
        self.n_iteration = iteration
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation

    def ant_colony_optimization(self, distances, n_ants, n_iterations, alpha, beta, evaporation):
        n_cities = self.distances.shape[0]
        pheromones = np.ones((n_cities, n_cities))
        best_path = None
        best_distance = float('inf')

        for _ in range(n_iterations):
            ant_paths = []
            for _ in range(n_ants):
                path = self.construct_path(self.distances, pheromones, self.alpha, self.beta)
                ant_paths.append(path)

            self.update_pheromones(pheromones, ant_paths, self.evaporation)

            current_best_path, current_best_distance = self.get_best_path(ant_paths, self.distances)
            if current_best_distance < best_distance:
                best_path = current_best_path
                best_distance = current_best_distance

        return best_path, best_distance

    def construct_path(self, distances, pheromones, alpha, beta):
        n_cities = self.distances.shape[0]
        current_city = np.random.randint(n_cities)
        unvisited_cities = set(range(n_cities))
        unvisited_cities.remove(current_city)
        path = [current_city]

        while unvisited_cities:
            next_city = self.select_next_city(current_city, unvisited_cities, self.distances, pheromones, self.alpha, self.beta)
            path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city

        return path

    def select_next_city(self, current_city, unvisited_cities, distances, pheromones, alpha, beta):
        pheromone_values = pheromones[current_city, list(unvisited_cities)]
        heuristic_values = 1.0 / (distances[current_city, list(unvisited_cities)] + 1e-6)
        probabilities = pheromone_values ** alpha * heuristic_values ** beta
        probabilities /= np.sum(probabilities)

        next_city = np.random.choice(list(unvisited_cities), p=probabilities)
        return next_city

    def update_pheromones(self, pheromones, ant_paths, evaporation):
        pheromones *= (1 - evaporation)
        for path in ant_paths:
            path_distance = self.calculate_path_distance(path, self.distances)
            for i in range(len(path) - 1):
                city_a, city_b = path[i], path[i + 1]
                pheromones[city_a, city_b] += 1.0 / path_distance

    def calculate_path_distance(self, path, distances):
        total_distance = 0
        for i in range(len(path) - 1):
            city_a, city_b = path[i], path[i + 1]
            total_distance += distances[city_a, city_b]

        return total_distance

    def get_best_path(self, ant_paths, distances):
        best_path = None
        best_distance = float('inf')
        for path in ant_paths:
            path_distance = self.calculate_path_distance(path, distances)
            if path_distance < best_distance:
                best_path = path
                best_distance = path_distance
        return best_path, best_distance
