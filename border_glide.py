import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

EPS = 1e-4


class FunctionWrapper:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return self.func(x)

    def reset(self):
        self.call_count = 0


# Целевая функция
def f(x):
    diff = x - np.array([14, 6])
    A = np.array([[2, 1], [1, 2]])
    return diff @ A @ diff.T


# Градиент целевой функции
def grad_f(x):
    return 2 * np.array([[2, 1], [1, 2]]) @ (x - np.array([14, 6]))


# Ограничения
def g1(x):
    return 0.5 * x[0] - x[1]


def g2(x):
    return x[0] ** 2 + x[1] ** 2 - 0.25 ** 2


def grad_penalty(x):
    grad = np.zeros(2)
    if g1(x) > 0:
        grad += 2 * g1(x) * np.array([0.5, -1])
    if g2(x) > 0:
        grad += 2 * g2(x) * np.array([2 * x[0], 2 * x[1]])
    return grad


def angle_between(v1, v2):
    """Возвращает угол в радианах между двумя векторами"""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Защита от численной ошибки
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def is_angle_small(f_grad, constraints_grads, threshold_rad=np.deg2rad(5)):
    """
    Проверяет, нет ли малого угла между антиградиентом и градиентами ограничений.
    """

    angle = angle_between(-f_grad, constraints_grads)
    if angle < threshold_rad:
        return True
    return False


# проекция на допустимое множество
def project_cvx(x):
    y = cp.Variable(2)
    obj = cp.Minimize(cp.sum_squares(y - x))  # формируем целевую функцию - квадрат расстояния до точки
    constr = [  # вводим ограничения - допустимое множество в задаче
        0.5 * y[0] - y[1] <= 0,
        y[0] ** 2 + y[1] ** 2 <= 0.25 ** 2
    ]
    prob = cp.Problem(obj, constr)  # формулируем задачу
    prob.solve()  # решаем задачу
    return y.value

def project_direction(x, grad):
    """
    Возвращает проекцию антиградиента на касательное пространство допустимого множества.
    Если ограничений нет, возвращает просто -grad.
    """
    A = []
    b = []

    # Активные ограничения (на границе)
    if abs(g1(x)) < EPS:
        A.append([0.5, -1])
        b.append(0)
    if abs(g2(x)) < EPS:
        A.append([2*x[0], 2*x[1]])
        b.append(0)

    A = np.array(A)
    b = np.array(b)

    if len(A) == 0:
        return -grad  # нет активных ограничений

    # Переменная: проецируемый вектор d
    d = cp.Variable(2)
    obj = cp.Minimize(cp.sum_squares(d + grad))  # хотим, чтобы d ≈ -grad
    constraints = [A @ d == b]  # d лежит в касательном пространстве
    prob = cp.Problem(obj, constraints)
    prob.solve()

    return d.value



def backtracking_line_search(f, grad_f, x, d, alpha=1.0, beta=0.5, sigma=1e-4):
    """
    метод осуществляет линейный поиск подходяшего шага, используя условие аармейко
    f       - целевая функция
    grad_f  - градиент целевой функции
    x       - текущая точка
    d       - направление спуска
    alpha   - начальный шаг
    beta    - множитель для уменьшения шага
    sigma   - параметр условия Аармейко (уровень "достаточности" снижения функции)
    """
    grad = grad_f(x)
    fx = f(x)
    iteration = 0
    # значение функции после шага / линейная аппроксимация функции с коэффициентом снижения сигма (от 0 до 1)
    while f(x + alpha * d) > fx + sigma * alpha * grad @ d:
        iteration += 1
        print(f"\nИтерация подбора шага: {iteration} ")
        print(
            f"Значение функции после шага: {f(x + alpha * d)}, Линейная аппроксимация функции: {fx + sigma * alpha * grad @ d}")
        print(f"Текущее alpha: {alpha}")
        print(x + alpha * d)
        plt.scatter((x + alpha * d)[0], (x + alpha * d)[1], color="green")
        alpha *= beta
        if alpha < 1e-10:
            return 0.0
    print(
        f"\nЗначение функции после шага: {f(x + alpha * d)}, Линейная аппроксимация функции: {fx + sigma * alpha * grad @ d}")
    print(f"Текущее alpha: {alpha}")
    return alpha


# Метод проекции градиента
def gradient_projection_method(x0, alpha_init=1.0, tol=1e-4, max_iter=1000):
    x = np.array(x0, dtype=float)
    history = []

    for iteration in range(max_iter):
        grad = grad_f(x)
        d = project_direction(x, grad)# Направление спуска

        # Линейный поиск с учётом направления d
        alpha = backtracking_line_search(func, grad_f, x, d, alpha=alpha_init)

        if alpha == 0:
            print("Шаг стал слишком малым")
            break

        x_new = project_cvx(x + alpha * d)
        fx = func(x_new)
        ang = 180- np.rad2deg(angle_between(grad_f(x_new), grad_penalty(x_new)))
        history.append((x_new.copy(), fx))

        print(f"Iter {iteration}: x = {x_new}, f(x) = {fx:.6f}, angle = {ang:.6f}")

        x = x_new
        if is_angle_small(grad_f(x), grad_penalty(x)):
            print("Сошлось!")
            break

    return x, history


# Запуск
func = FunctionWrapper(f)
start = [-0.2, 0]
solution, history = gradient_projection_method(start)

# Вывод результата
print("\nРезультат:")
print(f"Полученная точка: {solution}")
print(f"Значение целевой функции: {f(solution)}")
print(f"Число вызовов функции: {func.call_count}")

x_vals = np.linspace(-0.5, 1.5, 400)
y_vals = np.linspace(-0.5, 1.5, 400)

# Создаём сетку
X, Y = np.meshgrid(x_vals, y_vals)

# Вычисляем значения функции на сетке
Z = np.zeros_like(X)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        Z[j, i] = f([X[j, i], Y[j, i]])

plt.figure(figsize=(8, 8))
levels = np.logspace(np.log10(Z.min() + 1e-3), np.log10(Z.max()), 30)
contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
plt.clabel(contour, inline=True, fontsize=8)

# График траектории
x_hist = np.array([h[0] for h in history])

#plt.plot(x_hist[:, 0], x_hist[:, 1], '-o', label='Траектория')
plt.scatter(solution[0], solution[1], color='red', label='Решение')
plt.scatter(start[0], start[1], color='pink', label="Начальная точка")

# Ограничения
theta = np.linspace(0, 2 * np.pi, 100)
r = 0.25
x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)
plt.plot(x_circle, y_circle, 'g--', label='Ограничение: x² + y² ≤ 0.25²')

x_line = np.linspace(-0.3, 0.3, 100)
y_line = 0.5 * x_line
plt.plot(x_line, y_line, 'm--', label='Ограничение: y ≥ 0.5x')

x_last = x_hist[-1]
grad = grad_f(x_last)

# Нормализуем для наглядности
length = np.linalg.norm(grad)
if length > 0:
    grad_normalized = grad / length * 0.1
else:
    grad_normalized = grad

grad_p = grad_penalty(x_last)

length_p = np.linalg.norm(grad_p)
if length_p > 0:
    grad_normalized_p = grad_p / length_p * 0.1
else:
    grad_normalized_p = grad_p

plt.arrow(x_last[0], x_last[1], -grad_normalized[0], -grad_normalized[1],
          head_width=0.02, length_includes_head=True, color='red', label='Антиградиент')
plt.arrow(x_last[0], x_last[1], grad_normalized_p[0], grad_normalized_p[1],
          head_width=0.02, length_includes_head=True, color='blue', label='градиент')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Метод проекции градиента')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
