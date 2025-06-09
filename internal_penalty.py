import numpy as np
import matplotlib.pyplot as plt

x0_vec = np.array([14, 6])
A = np.array([[2, 1],
              [1, 2]])


class FunctionWrapper:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return self.func(x)

    def reset(self):
        self.call_count = 0


def f(x):
    diff = x - x0_vec
    return diff @ A @ diff.T  #2x**2-68x+632+2xy-52y+2y**2


# Градиент целевой функции
def grad_f(x):
    return 2 * A @ (x - x0_vec)


# Ограничения
def g1(x):
    return 0.5 * x[0] - x[1]


def g2(x):
    return x[0] ** 2 + x[1] ** 2 - 0.25 ** 2


# Логарифмическая барьерная функция
def barrier(x):
    return -np.log(-g1(x)) - np.log(-g2(x))


# Градиент барьерной функции
def grad_barrier(x):
    dg1 = np.array([0.5, -1.0])
    dg2 = np.array([2 * x[0], 2 * x[1]])
    return (-dg1 / (-g1(x))) + (-dg2 / (-g2(x)))


# Объединённая целевая функция
def target(x, r, f):
    return f(x) + r * barrier(x)


# Градиент объединённой целевой функции
def grad_F(x, r):
    return grad_f(x) + r * grad_barrier(x)


# Градиентный спуск
def gradient_descent(F, grad_F, x0, r, lr=0.001, max_iter=1000, eps=1e-4):
    x = x0.copy()
    print("Итерации градиентного спуска")
    print("\nIter |        x           |     F(x)       |  ||grad F(x)||  | g1    | g2  ")
    print("---------------------------------------------------------------")
    for i in range(max_iter):
        print("----")
        grad = grad_F(x, r)
        if np.linalg.norm(grad) < eps:
            break
            # Линейный поиск для сохранения допустимости
        step = lr
        while True:
            x_new = x - step * grad
            print(f"{i} |  {x_new}  | {f(x_new)} | {grad_F(x,r)}   | {g1(x_new)} | {g2(x_new)} | {r} | {step}")

            if g1(x_new) < 0 and g2(x_new) < 0:
                x = x_new
                break
            else:
                step *= 0.5  # уменьшаем шаг, пока не вернёмся в допустимую область
                if step < 1e-4:
                    print("Шаг стал слишком малым")
                    return x
    return x


# Метод внутренних штрафов
def interior_penalty_method(start):
    # Параметры
    r0 = 1.0
    beta = 0.5
    max_outer_iter = 100
    epsilon = 1e-4
    x_history = []

    # Начальная точка (должна быть строго внутри допустимой области)
    x = start
    print(f"Iter |          x         |   F(x)       |    ||grad F(x)||   | g1   | g2  | r")
    print(f" {0}  |  {x} | {func(x)}  | {grad_f((x))} | {g1(x)} | {g2(x)} | {r0}")
    for k in range(max_outer_iter):
        r = r0 * (beta ** k)
        x_new = gradient_descent(target, grad_F, x, r, lr=0.001)
        print("Итерация метода внутренних штрафов")
        print(f" {k+1}  |  {x_new} | {func(x_new)}  | {grad_f((x_new))} | {g1(x_new)} | {g2(x_new)} | {r}")
        x_history.append(x_new.copy())

        # Условие остановки: малое изменение x
        if np.linalg.norm(x_new - x) < epsilon:
            print(f"Сошлось за {k + 1} итераций.")
            break
        x = x_new

    return np.array(x_history)


# Запуск алгоритма
func = FunctionWrapper(f)
func.reset()
start=[-0.2,0]
x_hist = interior_penalty_method(start)

# Вывод результата
print("Приближённое решение:", x_hist[-1])
print("Найденный минимум функции: ", func(x_hist[-1]))
print("Число вызовов функции: ", func.call_count)

# График сходимости
plt.figure(figsize=(8, 8))
x_vals = np.linspace(-0.5, 1.5, 400)
y_vals = np.linspace(-0.5, 1.5, 400)

# Создаём сетку
X, Y = np.meshgrid(x_vals, y_vals)

# Вычисляем значения функции на сетке
Z = np.zeros_like(X)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        Z[j, i] = f([X[j, i], Y[j, i]])


levels = np.logspace(np.log10(Z.min() + 1e-3), np.log10(Z.max()), 30)
contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
plt.clabel(contour, inline=True, fontsize=8)

# Ограничения
theta = np.linspace(0, 2 * np.pi, 100)
r_circle = 0.25
x_circle = r_circle * np.cos(theta)
y_circle = r_circle * np.sin(theta)

# Траектория x_k
plt.plot(x_hist[:, 0], x_hist[:, 1], '-o', label='Траектория')
plt.scatter(x_hist[-1, 0], x_hist[-1, 1], color='red', label='Решение')
plt.scatter(start[0],start[1], color="pink", label="Начальная точка")

x_last = x_hist[-1]
grad = grad_f(x_last)

# Нормализуем для наглядности
length = np.linalg.norm(grad)
if length > 0:
    grad_normalized = grad / length * 0.1
else:
    grad_normalized = grad

plt.arrow(x_last[0], x_last[1], -grad_normalized[0], -grad_normalized[1],
          head_width=0.02, length_includes_head=True, color='red', label='Антиградиент')


plt.xlabel('x')
plt.ylabel('y')
plt.title('Траектория метода внутренних штрафов')
plt.legend()
plt.grid(True)



plt.plot(x_circle, y_circle, 'g--', label='Ограничение: x² + y² ≤ 0.25²')
x_line = np.linspace(-0.3, 0.3, 100)
y_line = 0.5 * x_line
plt.plot(x_line, y_line, 'm--', label='Ограничение: y ≥ 0.5x')
plt.axis('equal')
plt.legend()

plt.show()
