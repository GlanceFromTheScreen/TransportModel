from One_D_Problem_file import One_D_Problem
import matplotlib.pyplot as plt
import numpy as np

p1 = One_D_Problem()
accuracy = [0.1, 0.01, 0.001]

for item in accuracy:
    print(f'Точность: {item}')

    ans, n = p1.uniform_search_method(6, item)
    print('Метод равномерного поиска:\n', n, ' ', ((ans / item) // 1) * item, '+-', item / 2)
    ans, n = p1.trial_point_method(item)
    print('Метод пробных точек:\n', n, ' ', ((ans / item) // 1) * item, '+-', item / 2)
    ans, n = p1.golden_search(item)
    print('Метод золотого сечения:\n', n, ' ', ((ans / item) // 1) * item, '+-', item / 2)

    print()


x = np.linspace(p1.left_border, p1.right_border, 100)
y = []
for i in range(100):
    y.append(p1.target_function(x[i]))

plt.plot(x, y, label='target function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.show()



