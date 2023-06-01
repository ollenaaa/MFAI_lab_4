import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
from ant_colony import AntColony


def generate_distance_matrix(N, distance_range):
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distance_matrix[i][j] = distance_matrix[j][i] = random.randint(distance_range[0], distance_range[1])
    return distance_matrix


def save_distance_matrix(distance_matrix, filename):
    np.savetxt(filename, distance_matrix, fmt='%d')


def load_distance_matrix(filename):
    return np.loadtxt(filename, dtype=int)


N = random.randint(25, 35)
distance_range = (10, 100)
filename = 'distance_matrix.txt'

# distance_matrix = generate_distance_matrix(N, distance_range)
# save_distance_matrix(distance_matrix, filename)

distance_matrix = load_distance_matrix(filename)

ant_colony = AntColony(distance_matrix, 62, 200, 0.95, 2.0, 0.9)
shortest_path, shortest_dist = ant_colony.ant_colony_optimization(distance_matrix, 35, 200, 0.95, 2.0, 0.5)
print ("shorted_path: {}".format(shortest_path), "\nshorted_dist: {}".format(shortest_dist))


radius = 20
cities = []
for i in range(len(distance_matrix)):
    angle = i * 2 * np.pi / len(distance_matrix)
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    cities.append((x, y))

cities = np.array(cities)
path_cities = cities[shortest_path]

plt.figure(figsize=(6, 6))
plt.scatter(cities[:, 0], cities[:, 1], color='red', label='Cities')
plt.xlim(-radius - 1, radius + 1)
plt.ylim(-radius - 1, radius + 1)
plt.gca().set_aspect('equal', adjustable='box')

for i, city in enumerate(cities):
    plt.text(city[0], city[1], str(i), color='black')

for i in range(len(shortest_path) - 1):
    city_a = path_cities[i]
    city_b = path_cities[i + 1]
    distance = distance_matrix[shortest_path[i], shortest_path[i + 1]]
    plt.arrow(city_a[0], city_a[1], city_b[0] - city_a[0], city_b[1] - city_a[1], length_includes_head=True, head_width=0.2, color='blue')
    # plt.annotate(str(distance), ((city_a[0] + city_b[0]) / 2, (city_a[1] + city_b[1]) / 2), color='blue')

plt.legend()
plt.show()