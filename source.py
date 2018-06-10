import numpy as np
import math as mt
import random as rd
import matplotlib.pyplot as plt


def calculate_distance_error(true_distance, calculated_distance):
    differences = calculated_distance - true_distance
    differences_to_sum = np.triu(np.matrix(differences))
    squared_differences = []
    for i in range(len(differences_to_sum)):
        squared_differences.append((differences_to_sum[i] @ differences_to_sum[i].T) ** (1/2))
    return np.sum(squared_differences)


def calculate_distance_to_all_points_by_point_index(i, matrix):
    return calculate_distance_to_all_points(matrix[i], matrix)


def calculate_distance_to_all_points(point, matrix):
    diff_matrix = matrix - point
    diff_vector = []
    for i in range(len(diff_matrix)):
        diff_vector.append((diff_matrix[i] @ diff_matrix[i].T) ** (1/2))
    return np.array(diff_vector)


def calculate_distance(point_a, point_b):
    return calculate_squared_distance(point_a, point_b) ** (1/2)


def calculate_squared_distance(point_a, point_b):
    diff_vector = calculate_distance_vector(point_a, point_b)
    return diff_vector @ diff_vector.T


def calculate_distance_vector(point_a, point_b):
    return point_a - point_b


def read_distance_matrix_from_file(path, elements_separator, lines_separator):
    file = open(path, 'r')
    lines = [line.replace("\n", "").split(lines_separator) for line in file][0]
    strings_matrix = [line.split(elements_separator) for line in lines]
    matrix = [[float(element) for element in line] for line in strings_matrix]
    return np.array(matrix)


def get_learning_rate(iteration: int):
    return 1 / mt.sqrt(iteration)


def is_last_iteration(previous_result: float, current_result: float, epsilon: float):
    return abs(1 - (current_result / previous_result)) < epsilon


def calculate_next_points_vector(current_points: np.array, gradient: np.array, learning_rate: float):
    return current_points - learning_rate * gradient


def calculate_distance_matrix(points: np.array):
    distance_matrix = []
    for point in points:
        distance_matrix.append(calculate_distance_to_all_points(point, points))
    return np.array(distance_matrix)


def calculate_gradient_element(true_distances: np.array, points: np.array, i: int, points_amount: int, k: int):
    element = 0

    for j in range(i + 1, points_amount):
        dif = points[i] - points[j]
        squared_dif = dif ** 2
        val_squared_dif = squared_dif @ squared_dif.T

        element += 2 * (np.sqrt(val_squared_dif) - true_distances[i, j]) * val_squared_dif * (points[i, k] - points[j, k])

        #distance = calculate_distance(points[i], points[j])
        #squared_distance = calculate_squared_distance(points[i], points[j])
        #sum_of_distance_vector = np.sum(calculate_distance_vector(points[i], points[j]))

        #element += 2 * (np.sqrt(points[i,k] - points[j, k] ** 2) - true_distances[i,j]) * (points[i,k] - points[j, k] ** 2) * (points[i,k] - points[j, k])
        #element += 2 * (distance - true_distances[i, j]) * squared_distance * sum_of_distance_vector

    return element


def calculate_gradient(points: np.array, true_distances: np.array):
    gradient = np.zeros(points.shape)

    for i in range(gradient.shape[0]):
        for k in range(gradient.shape[1]):
            gradient[i, k] = calculate_gradient_element(true_distances, points, i, gradient.shape[0], k)

    return gradient


def get_random_start_points(amount_of_points: int,
                            min_x_coordinate: int, min_y_coordinate: int,
                            max_x_coordinate: int, max_y_coordinate: int):
    points = []
    for i in range(amount_of_points):
        points.append([rd.uniform(min_x_coordinate, max_x_coordinate), rd.uniform(min_y_coordinate, max_y_coordinate)])
    return np.array(points)


def do_optimization(start_points: np.array, true_distances: np.array, iteration: int, epsilon: float,
                    max_steps_without_improvement: int, optimize_alpha: bool, use_numeric_gradient: bool):

    learning_rate = get_learning_rate(iteration)

    current_error_matrix = calculate_distance_matrix(start_points)
    current_error_value = calculate_distance_error(true_distances, current_error_matrix)
    current_points = start_points
    steps_without_improvement = 0

    best_points = current_points
    best_error = current_error_value

    while True:
        previous_error_value = current_error_value
        if use_numeric_gradient:
            gradient = calculate_gradient_numeric(current_points, 0.000000000001, true_distances,
                                                          current_error_value)
        else:
            gradient = calculate_gradient(current_points, true_distances)

        if optimize_alpha:
            learning_rate = get_optimal_alfa(gradient, current_points, true_distances, epsilon, 0.000001, 3)

        next_points = calculate_next_points_vector(current_points, gradient, learning_rate)
        current_error_matrix = calculate_distance_matrix(next_points)
        current_error_value = calculate_distance_error(true_distances, current_error_matrix)
        current_points = next_points

        if current_error_value < previous_error_value:
            best_error = current_error_value
            best_points = current_points
            #steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if is_last_iteration(previous_error_value, current_error_value, epsilon) \
                or steps_without_improvement > max_steps_without_improvement:
            print("Błąd wynosi: " + str(current_error_value) + " abort")
            break
        else:
            print("Błąd wynosi: " + str(current_error_value) + " wykonuję kolejną iterację")

    return [best_error, best_points]


def optimize_distance(file_name, start_iteration, end_iteration, step, title, use_numeric_gradient, labels):
    elements_separator = ' '
    lines_separator = ';'
    matrix = read_distance_matrix_from_file(file_name, elements_separator, lines_separator)

    best_error = 1000000000
    best_points = []
    best_i = 0

    for j in range(100):
        start_points = get_random_start_points(matrix.shape[0], 0, 0, matrix.max(), matrix.max())
        for i in range(start_iteration, end_iteration, step):
            print("Start: " + str(i))
            tuple = do_optimization(start_points, matrix, i, 0.0000001, 10, False, use_numeric_gradient)
            if tuple[0] < best_error:
                best_error = tuple[0]
                best_points = tuple[1]
                best_i = i

    print(best_error)
    print(best_points)
    print(best_i)

    visualize(best_points, title, labels)


def optimize_distance_with_optimal_alfa(file_name, title, use_numeric_gradient, labels):
    elements_separator = ' '
    lines_separator = ';'
    matrix = read_distance_matrix_from_file(file_name, elements_separator, lines_separator)

    best_error = 1000000000
    best_points = []

    for j in range(100):
        start_points = get_random_start_points(matrix.shape[0], 0, 0, matrix.max(), matrix.max())
        tuple = do_optimization(start_points, matrix, 1, 0.0000001, 10, True, use_numeric_gradient)
        if tuple[0] < best_error:
            best_error = tuple[0]
            best_points = tuple[1]

    print(best_error)
    print(best_points)

    visualize(best_points, title, labels)


def calculate_gradient_element_numeric(points: np.array, epsilon: float, true_distance: np.array,
                                       i: int, j: int, current_error: float):
    points[i, j] += epsilon
    new_distances = calculate_distance_matrix(points)
    new_error = calculate_distance_error(true_distance, new_distances)
    return (new_error - current_error) / epsilon


def calculate_gradient_numeric(points: np.array, epsilon: float, true_distance: np.array, current_error: float):
    gradient = np.zeros(points.shape)

    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):
            gradient[i, j] = calculate_gradient_element_numeric(points, epsilon, true_distance, i, j, current_error)

    return gradient


def calculate_numeric_gradient_for_alfa(points: np.array, epsilon: float, true_distance: np.array, current_error: float,
                                        points_gradient: np.array, alfa: float):
    new_points = calculate_next_points_vector(points, points_gradient, alfa + epsilon)
    new_distances = calculate_distance_matrix(new_points)
    new_value = calculate_distance_error(true_distance, new_distances)
    return (new_value - current_error) / epsilon


def get_optimal_alfa(points_gradient: np.array, points: np.array, true_distances: np.array,
                     epsilon: float, learning_rate: float, max_steps_without_improvement: float):
    alfa = rd.uniform(0.000001, 0.09)

    new_points = calculate_next_points_vector(points, points_gradient, alfa)
    new_distances = calculate_distance_matrix(new_points)

    value = calculate_distance_error(true_distances, new_distances)

    best_error = value
    best_alfa = alfa
    steps_without_improvement = 0
    max_deepth = 100

    while True:
        max_deepth -= 1
        alfa_gradient = calculate_numeric_gradient_for_alfa(points, 0.001, true_distances, value, points_gradient,
                                                            alfa)
        new_alfa = alfa - learning_rate * alfa_gradient
        new_points = calculate_next_points_vector(points, points_gradient, new_alfa)
        new_distances = calculate_distance_matrix(new_points)

        new_value = calculate_distance_error(true_distances, new_distances)

        alfa = new_alfa
        print("ALFA: " + str(alfa))

        if value < best_error:
            best_error = value
            best_alfa = new_alfa
        else:
            steps_without_improvement += 1
        if max_deepth == 0 or is_last_iteration(value, new_value, 0.0001) or steps_without_improvement > max_steps_without_improvement:
            break

        value = new_value

    print("BEST ALFA: " + str(best_alfa))
    return best_alfa


def visualize(points: np.array, title, labels):

    if labels is not None:
        for i in range(len(points)):
            plt.plot(points[i,0], points[i,1], 'o', label = labels[i])
        plt.legend(loc='best')

    else:
        plt.title(title)
        plt.plot(points.T[0], points.T[1], 'ro')
    plt.show()


def main():
    #optimize_distance_with_optimal_alfa("Resources/triangle", "Gradient numeryczny - metoda najszybszego spadku", True, None)
    #optimize_distance("Resources/triangle", 100, 101, 1, "Gradient numeryczny - metoda spadku wzdłuż gradientu", True, None)
    #optimize_distance("Resources/line", 100, 101, 1, "Gradient numeryczny - metoda spadku wzdłuż gradientu", True, None)
    #optimize_distance_with_optimal_alfa("Resources/line", "Gradient numeryczny - metoda najszybszego spadku", True, None)
    #optimize_distance_with_optimal_alfa("Resources/triangle", "Gradient analityczny - metoda najszybszego spadku", False, None)
    #optimize_distance("Resources/triangle", 100, 101, 1, "Gradient analityczny - metoda spadku wzdłuż gradientu", False, None)
    #optimize_distance("Resources/line", 100, 101, 1, "Gradient analityczny - metoda spadku wzdłuż gradientu", False, None)
    #optimize_distance_with_optimal_alfa("Resources/line", "Gradient analityczny - metoda najszybszego spadku", False, None)
    cities = ['Warszawa','Kraków','Łódź','Wrocław','Poznań','Gdańsk','Szczecin','Bydgoszcz','Lublin','Katowice']
    optimize_distance("Resources/cities", 2, 3, 1, "Gradient numeryczny - metoda spadku wzdłuż gradientu", True, cities)


if __name__ == '__main__':
    main()
