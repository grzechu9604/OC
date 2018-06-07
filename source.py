import numpy as np


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
    arr = np.array([point_a, point_b, 2 * point_a, 10 * point_a, 3 * point_a])
    print(calculate_distance_to_all_points(point_b, arr))
    print(calculate_distance_to_all_points_by_point_index(1, arr))


if __name__ == '__main__':
    main()
