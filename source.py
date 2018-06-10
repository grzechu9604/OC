import numpy as np
import math as mt
import random as rd
import matplotlib.pyplot as plt


def calculate_distance_error(true_differences, calculated_differences):
    differences = calculated_differences - true_differences
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
    return np.absolute(point_a - point_b)


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
        distance = calculate_distance(points[i], points[j])
        squared_distance = calculate_squared_distance(points[i], points[j])
        sum_of_distance_vector = np.sum(calculate_distance_vector(points[i], points[j]))

        element += 2 * (distance - true_distances[i, j]) * squared_distance * sum_of_distance_vector

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
                    max_steps_without_improvement: int):

    learning_rate = get_learning_rate(iteration)

    current_error_matrix = calculate_distance_matrix(start_points)
    current_error_value = calculate_distance_error(true_distances, current_error_matrix)
    current_points = start_points
    steps_without_improvement = 0

    best_points = current_points
    best_error = current_error_value

    while True:
        previous_error_value = current_error_value
        gradient = calculate_gradient(current_points, true_distances)

        numeric_gradient = calculate_gradient_numeric(current_points, 0.0000001, true_distances, current_error_value)

        next_points = calculate_next_points_vector(current_points, numeric_gradient, learning_rate)
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


def optimize_distance(file_name, start_iteration, end_iteration):
    elements_separator = ' '
    lines_separator = ';'
    matrix = read_distance_matrix_from_file(file_name, elements_separator, lines_separator)

    best_error = 1000000000
    best_points = []
    best_i = 0

    for j in range(100):
        start_points = get_random_start_points(matrix.shape[0], 0, 0, matrix.max(), matrix.max())
        for i in range(start_iteration, end_iteration):
            print("Start: " + str(i))
            tuple = do_optimization(start_points, matrix, i, 0.0000001, 10)
            if tuple[0] < best_error:
                best_error = tuple[0]
                best_points = tuple[1]
                best_i = i

    print(best_error)
    print(best_points)
    print(best_i)

    plt.plot(best_points.T[0], best_points.T[1], 'ro')
    plt.show()


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


def main():
    optimize_distance("Resources/triangle", 1000, 1001)
    #optimize_distance("Resources/line", 100000, 100002)
    #optimize_distance("Resources/cities", 20003550605550, 20003550605553)


if __name__ == '__main__':
    main()
