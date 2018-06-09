import numpy as np
import math as mt
import random as rd


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
    diff_vector = np.absolute(point_a - point_b)
    distance = (diff_vector @ diff_vector.T) ** (1/2)
    return distance


def read_distance_matrix_from_file(path, elements_separator, lines_separator):
    file = open(path, 'r')
    lines = [line.replace("\n", "").split(lines_separator) for line in file][0]
    strings_matrix = [line.split(elements_separator) for line in lines]
    matrix = [[float(element) for element in line] for line in strings_matrix]
    return np.array(matrix)


def get_learning_rate(iteration: int):
    return 1 / mt.sqrt(iteration)


def is_last_iteration(previous_result: float, current_result: float, epsilon: float):
    return 1 - (current_result / previous_result) < epsilon


def calculate_next_points_vector(current_points: np.array, gradient: np.array, learning_rate: float):
    return current_points - learning_rate * gradient


def calculate_distance_matrix(points: np.array):
    distance_matrix = []
    for point in points:
        distance_matrix.append(calculate_distance_to_all_points(point, points))
    return np.array(distance_matrix)


def calculate_gradient(current_points: np.array):
    # TODO implementacja gradientu
    return current_points


def get_random_start_points(amount_of_points: int,
                            min_x_coordinate: int, min_y_coordinate: int,
                            max_x_coordinate: int, max_y_coordinate: int):
    points = []
    for i in range(amount_of_points):
        points.append([rd.randint(min_x_coordinate, max_x_coordinate), rd.randint(min_y_coordinate, max_y_coordinate)])
    return np.array(points)


def do_optimization(start_points: np.array, true_distances: np.append, iteration: int, epsilon: float,
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
        gradient = calculate_gradient(current_points)
        next_points = calculate_next_points_vector(current_points, gradient, learning_rate)
        current_error_matrix = calculate_distance_matrix(next_points)
        current_error_value = calculate_distance_error(true_distances, current_error_matrix)

        if current_error_value < previous_error_value:
            best_error = current_error_value
            best_points = current_points
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

        if is_last_iteration(previous_error_value, current_error_value, epsilon) \
                or steps_without_improvement > max_steps_without_improvement:
            break
        else:
            print("Błąd wynosi: " + current_error_value + " wykonuję kolejną iterację")

    return [best_error, best_points]


def main():
    elements_separator = ' '
    lines_separator = ';'
    matrix = read_distance_matrix_from_file("Resources/triangle", elements_separator, lines_separator)
    point_a = np.array([1, 1, 1])
    point_b = np.array([0, 0, 0])
    print(calculate_distance(point_a, point_b))
    print(calculate_distance(point_b, point_a))
    print(calculate_distance(2 * point_a, point_b))
    print(calculate_distance(10 * point_a, point_b))
    print(calculate_distance(3 * point_a, point_b))
    arr = np.array([point_a, point_b, 2 * point_a, 5 * point_a, 6 * point_a, 7 * point_a])

    print("TU:")
    print(calculate_distance_to_all_points_by_point_index(0, arr))
    print(calculate_distance_to_all_points_by_point_index(1, arr))
    print(calculate_distance_to_all_points_by_point_index(2, arr))
    print(calculate_distance_to_all_points_by_point_index(3, arr))
    print(calculate_distance_to_all_points_by_point_index(4, arr))
    print(calculate_distance_to_all_points_by_point_index(5, arr))
    print("TU:")
    print(calculate_distance_matrix(arr))

    print(calculate_distance_error(arr, arr))
    print(calculate_distance_error(arr, 4 * arr))

    print(calculate_next_points_vector(matrix, matrix, 1))
    print(calculate_next_points_vector(matrix, matrix, 0.5))
    print(calculate_next_points_vector(matrix, matrix, 0.1))

    print(get_random_start_points(5, 1, 1, 8, 8))


if __name__ == '__main__':
    main()
