import numpy as np


def calculate_distance(point_a, point_b):
    distance: float = 0
    for i in range(len(point_a)):
        distance += (point_a[i] - point_b[i]) ** 2
    distance **= 1/2
    return distance


def read_distance_matrix_from_file(path, elements_separator, lines_separator):
    file = open(path, 'r')
    lines = [line.replace("\n","").split(lines_separator) for line in file][0]
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
    print(calculate_distance(2 * point_a, point_b))
    print(calculate_distance(10 * point_a, point_b))
    print(calculate_distance(3 * point_a, point_b))


if __name__ == '__main__':
    main()

