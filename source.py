import numpy as np


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


if __name__ == '__main__':
    main()

