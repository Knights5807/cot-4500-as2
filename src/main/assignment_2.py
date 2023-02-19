import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)


########## Question 1
def q1_neville_method(give, x, y):
    n = len(x)
    m = len(y)
    p = [[0.0 for column in range(n)] for row in range(n)]

    for row in range(n):
        p[row][0] = y[row]

    for row in range(1, n):
        for column in range(1, row + 1):
            p[row][column] = ((give - x[row - column]) * p[row][column - 1] - (give - x[row]) * p[row - 1][
                column - 1]) / (
                                     x[row] - x[row - column])

    return p[n - 1][n - 1]


give = 3.7
x = (3.6, 3.8, 3.9)
y = (1.675, 1.436, 1.318)
print(q1_neville_method(give, x, y))
print("")


########## Question 2
def newton_forward_difference(x, y):
    n = len(x)
    Y = np.zeros((n, n))
    Y[:, 0] = y
    for column in range(1, n):
        for row in range(column, n):
            Y[row, column] = (Y[row, column - 1] - Y[row - 1, column - 1]) / (x[row] - x[row - column])
    return Y[-1, :]


x = [7.2, 7.4, 7.5, 7.6]
y = [23.5492, 25.3913, 26.8224, 27.4589]

first_degree = newton_forward_difference(x[:2], y[:2])
second_degree = newton_forward_difference(x[:3], y[:3])
third_degree = newton_forward_difference(x, y)

print([first_degree[1], second_degree[2], third_degree[3]])
print("")


########## Question 3: Print the 3rd degree approximation of f(7.3)
def q3_newton_forward(give, x, y):
    length = len(x)
    z = np.zeros((length, length))
    z[:, 0] = y
    for column in range(1, length):
        for row in range(column, length):
            z[row, column] = (z[row, column - 1] - z[row - 1, column - 1]) / (x[row] - x[row - column])
    result = y[0]
    for row in range(1, length):
        one = 1
        for column in range(row):
            one *= (give - x[column])
        result += one * z[row, row]
    return result


give = 7.3
x = (7.2, 7.4, 7.5, 7.6)
y = (23.5492, 25.3913, 26.8224, 27.4589)
print(q3_newton_forward(give, x, y))
print("")


########## Question 4:
import numpy as np


def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # get left cell entry
            left: float = matrix[i][j - 1]
            # get diagonal left entry
            diagonal_left: float = matrix[i - 1][j - 1]
            # order of numerator is SPECIFIC.
            numerator: float = (left - diagonal_left)
            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i - j + 1][0]
            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation

    return matrix


def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    # matrix size changes because of "doubling" up info for hermite
    num_of_points = len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))
    # populate x values (make sure to fill every TWO rows)
    for i, x in enumerate(x_points):
        matrix[2 * i][0] = x
        matrix[2 * i + 1][0] = x

    # pre-populate y values (make sure to fill every TWO rows)
    for i, y in enumerate(y_points):
        matrix[2 * i][1] = y
        matrix[2 * i + 1][1] = y

    # pre-populate with derivatives (make sure to fill every TWO rows. starting row changes)
    for i, slope in enumerate(slopes):
        matrix[2 * i + 1][2] = slope

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)


hermite_interpolation()

########## Question 5
x = [2, 5, 8, 10]
y = [3, 5, 7, 9]
length = len(x)
z = np.zeros(length - 1)
d = np.zeros(length)
for row in range(length - 1):
    z[row] = x[row + 1] - x[row]
    d[row + 1] = (y[row + 1] - y[row]) / z[row]

A = np.zeros((length, length))
b = np.zeros(length)
for row in range(1, length - 1):
    A[row][row - 1] = z[row - 1]
    A[row][row] = 2 * (z[row - 1] + z[row])
    A[row][row + 1] = z[row]
    b[row] = 3 * (d[row + 1] - d[row])

A[0][0] = 1
A[length - 1][length - 1] = 1

x = np.linalg.solve(A, b)
print("")
print(A)
print("")
print(b)
print("")
print(x)

