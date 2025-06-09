import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-4

# Обертка для подсчета числа вызовов функции
class FunctionWrapper:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return self.func(x)

    def reset(self):
        self.call_count = 0


# Целевая функция и ее градиент
x0_vec = np.array([14, 6])
A = np.array([[2, 1],
              [1, 2]])


def f(x):
    diff = x - x0_vec
    return diff @ A @ diff.T  #2x**2-68x+632+2xy-52y+2y**2


def grad_f(x):
    return 2 * A @ (x - x0_vec)


# Ограничения и их градиенты
r = (14 + 6 + 5) / 100  # = 0.25


def g1(x):
    return 0.5 * x[0] - x[1]


def g2(x):
    return x[0] ** 2 + x[1] ** 2 - r ** 2


def penalty(x):
    return max(0, g1(x)) ** 2 + max(0 , g2(x)) ** 2


def grad_penalty(x):
    grad = np.zeros(2)
    if g1(x) > 0:
        grad += 2 * g1(x) * np.array([0.5, -1])
    if g2(x) > 0:
        grad += 2 * g2(x) * np.array([2 * x[0], 2 * x[1]])
    return grad


# Градиентный спуск
def gradient_descent_ext(f, grad, x0, alpha=0.001, tol=0.0001, max_iter=1000):
    x = np.array(x0, dtype=float)
    history = []
    #print("\nIter |        x           |     f(x)       |  Модуль градиента  ")
    #print("---------------------------------------------------------------")
    for iteration in range(max_iter):
        grad_val = grad(x)
        step_size = np.linalg.norm(grad_val)
        fv = f(x)

        #print(f"{iteration}   | {x} | {fv} | {step_size}  ")
        history.append((x.copy(), fv))

        if step_size < tol:
            break
        x -= alpha * grad_val

    return x, history


# Метод внешних штрафов
def external_penalty_method(x0, wrapped_f, R0=1.0, c=2, max_outer=20):
    x = np.array(x0, dtype=float)
    phi = R0
    last_callcount = wrapped_f.call_count

    # Для построения графиков
    f_values = []
    p_values = []
    x_history = [x.copy()]
    print(
        f"Iter | x         |        y  | Значение функции |  Значение функции ограничений | Значение первого ограничения | Значение второго ограничения | r")

    print(f"{0:>4} | {x0[0]:>8.5f}, {x0[1]:>8.5f} |"
          f" {wrapped_f(x0):>10.5f} |       {penalty(x0):>9.2e} |         {g1(x0)} |       {g2(x0)}  |   {phi}")

    for outer_iter in range(max_outer):
        def penalized(x): return wrapped_f(x) + phi * penalty(x)
        def grad_penalized(x): return grad_f(x) + phi * grad_penalty(x)

        last_callcount = wrapped_f.call_count
        x_new, grad_hist = gradient_descent_ext(penalized, grad_penalized, x, alpha=0.001, tol=0.0001)
        if outer_iter ==0:
            grad1_hist = grad_hist
        fx = wrapped_f(x_new)
        px = penalty(x_new)
        print("-------------------------------------------------------")
        print(f"Iter |    x    |       y      | Значение функции |  Значение функции ограничений | Значение первого ограничения | Значение второго ограничения | r")

        print(f"{outer_iter+1:>4} | {x_new[0]:>8.5f}, {x_new[1]:>8.5f} |"
              f" {fx:>10.5f} | {px:>9.2e} | {g1(x_new)} | {g2(x_new)}  |   {phi}")

        f_values.append(fx)
        p_values.append(px)
        x_history.append(x_new.copy())

        if px < EPS:
            print("Ограничения выполнены с заданной точностью.")
            break

        x = x_new
        phi *= c

    return np.array(x_history), np.array(f_values), np.array(p_values), wrapped_f.call_count, grad1_hist


# Запуск
func = FunctionWrapper(f)
func.reset()
start=[-1,-1]
x_history, f_values, p_values, call_count, grad1_hist = external_penalty_method(start, func)


print(f"Получена точка: {x_history[-1]}, со значением функции: {f(x_history[-1])}, "
      f"с общим числом вызовов функции: {call_count}")

# График траектории точки x_k
plt.figure(figsize=(8, 8))
x_vals = np.linspace(-1, 2.5, 400)
y_vals = np.linspace(-1, 2.5, 400)

# Создаём сетку
X, Y = np.meshgrid(x_vals, y_vals)

# Вычисляем значения функции на сетке
Z = np.zeros_like(X)
for i in range(len(x_vals)):
    for j in range(len(y_vals)):
        Z[j, i] = f([X[j, i], Y[j, i]])

plt.plot(grad1_hist[0][0][0], grad1_hist[0][0][1], "-o", label="grad", color="black")
for i in range(len(grad1_hist)):
    plt.plot(grad1_hist[i][0][0], grad1_hist[i][0][1], "-o", color="black")


levels = np.logspace(np.log10(Z.min() + 1e-3), np.log10(Z.max()), 30)
contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
plt.clabel(contour, inline=True, fontsize=8)
plt.plot(x_history[:, 0], x_history[:, 1], '-o', label='Траектория x_k')

plt.scatter(x_history[-1, 0], x_history[-1, 1], color='yellow', label='Решение', zorder=5)
plt.scatter(start[0], start[1], color='pink', label="Начальная точка")


x_last = x_history[-1]
grad = grad_f(x_last)

# Нормализуем для наглядности
length = np.linalg.norm(grad)
if length > 0:
    grad_normalized = grad / length * 0.1
else:
    grad_normalized = grad


# Ограничения
r_circle = 0.25
theta = np.linspace(0, 2*np.pi, 100)
x_circle = r_circle * np.cos(theta)
y_circle = r_circle * np.sin(theta)
plt.plot(x_circle, y_circle, 'g--', label='x² + y² = 0.25²')

x_line = np.linspace(-0.5, 1.5, 100)
y_line = 0.5 * x_line
plt.plot(x_line, y_line, 'm--', label='y = 0.5x')

plt.arrow(x_last[0], x_last[1], -grad_normalized[0], -grad_normalized[1],
          head_width=0.02, length_includes_head=True, color='red', label='Антиградиент', zorder=10)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Траектория метода внешних штрафов')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
