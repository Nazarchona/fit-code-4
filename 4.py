import numpy as np

# Задача 1: Система лінійних рівнянь зі скріншоту 3 (приклад під номером 7)
A = np.array([[1, -1, 2], [3, 2, -1], [2, 3, 1]])
B = np.array([2, 4, 5])

# Метод Крамера
det_A = np.linalg.det(A)

# Визначники для чисельників
det_A1 = np.linalg.det(np.column_stack((B, A[:, 1], A[:, 2])))
det_A2 = np.linalg.det(np.column_stack((A[:, 0], B, A[:, 2])))
det_A3 = np.linalg.det(np.column_stack((A[:, 0], A[:, 1], B)))

# Розв'язок методом Крамера
x1_cramer = det_A1 / det_A
x2_cramer = det_A2 / det_A
x3_cramer = det_A3 / det_A

# Матричний метод (x = A^(-1) * B)
A_inv = np.linalg.inv(A)
X_matrix = np.dot(A_inv, B)

# Метод Гауса (через вирішення системи лінійних рівнянь)
X_gauss = np.linalg.solve(A, B)

# Результати
det_A, X_matrix, X_gauss, (x1_cramer, x2_cramer, x3_cramer)
