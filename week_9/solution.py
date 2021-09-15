from sys import stdin
from copy import deepcopy
import numpy as np


class MatrixError(BaseException):

    def __init__(self, matrix1, matrix2):
        self.matrix1 = Matrix(matrix1)
        self.matrix2 = Matrix(matrix2)


class Matrix:

    def __init__(self, matrix):
        self.matrix = deepcopy(matrix)

    def __add__(self, other):
        if self.size() == other.size():
            return Matrix([[other.matrix[i][j] + self.matrix[i][j] for j in
                            range(len(self.matrix[0]))] for i in
                           range(len(self.matrix))])
        else:
            raise MatrixError(self.matrix, other.matrix)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            if self.size()[1] == other.size()[0]:
                res = Matrix([[0 for _ in range(other.size()[1])] for _ in
                              range(self.size()[0])])
                for i in range(len(self.matrix)):
                    for j in range(len(other.matrix[0])):
                        for k in range(len(other.matrix)):
                            res.matrix[i][j] += self.matrix[i][k] * \
                                                other.matrix[k][j]
                return res
            else:
                raise MatrixError(self.matrix, other.matrix)
        elif isinstance(other, int) or isinstance(other, float):
            return Matrix([[col * other for col in row] for row in
                           self.matrix])

    def __str__(self):
        res = ''
        for row in self.matrix:
            res += '\t'.join(map(str, row)) + '\n'
        return res.strip()

    def size(self):
        return (len(self.matrix), len(self.matrix[0]))

    __rmul__ = __mul__

    def solve(self, vector):
        try:
            x = np.linalg.solve(np.array(self.matrix), np.array(vector))
            return x
        except 'LinAlgError':
            raise MatrixError(self.matrix, vector)

    def transpose(self):
        transMatrix = list(zip(*self.matrix))
        self.matrix = transMatrix
        return Matrix(transMatrix)

    @staticmethod
    def transposed(matrix):
        transMatrix = list(zip(*matrix.matrix))
        return Matrix(transMatrix)


class SquareMatrix(Matrix):
    def __pow__(self, power, modulo=None):
        return Matrix(np.linalg.matrix_power(self.matrix, power))


exec(stdin.read())
