import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
complexity_rate = 0


def f(x):
    global complexity_rate
    complexity_rate += 1
    return np.exp(-0.01 * 22 * x) + np.exp(0.01 * 14 * x) + 1 * x ** 2



def get_short_hex_output(decimal_number, num_of_syms=5):
    hex_number = decimal_number.hex().split('x')[1]
    sign = decimal_number.hex()[0]
    if sign == '-':
        s = sign + hex_number[:num_of_syms + 2] + '...p' + hex_number.split('p')[1]
    else:
        s = hex_number[:num_of_syms + 2] + '...p' + hex_number.split('p')[1]
    return s


# То, что Михеев сказал скинуть
def directional_search(f, initial_dot, step):
    global complexity_rate

    # complexity_rate - счётчик вызовов функции
    complexity_rate = 0

    print('Метод направленного поиска')
    # directional_df - таблица, соответствующая поиску триады
    directional_df = pd.DataFrame(columns=['x1', 'f1', 'x2', 'f2',
                                           'x3', 'f3', 'Кол-во вызовов функции'])

    # Комментарий не Михеева: python будет ругаться на попытку сконкатенировать строку с пустым датафреймом
    # на первой итерации, но всё равно отработает в текущей версии pandas. Право исправить предоставляется читателю.
    # Шаг step выбираем самостоятельно
    # n определяется, исходя из значения минимума ВАШЕЙ задачи!
    # Конец комментария не Михеева
    n = -5

    # Если step > 0, то x1 < x2 < x3. Иначе - x1 > x2 > x3
    x1 = initial_dot
    x2 = x1 + step

    f1 = f(x1)
    f2 = f(x2)

    # Ниже следующим условием можно развернуть поиск триады в обратном направлении (в цикле -- см. ниже)
    # Это позволяет в цикле заниматься поиском триады с постоянным шагом как по знаку, так и по величине
    if f1 < f2:
        step = -step
        temp = x1
        x1 = x2
        x2 = temp

        temp = f1
        f1 = f2
        f2 = temp

    print('1) Поиск триады')
    # Цикл поиска триады. Завершается, когда триада найдена
    # x1 и f1 для вычисления внутри цикла не применяются, они присутствуют для сохранения в памяти третьего узла
    # и значения целевой функции в нём (для дальнейших вычислений внутри триады в следующем цикле -- в цикле работы
    # с триадой)
    while True:
        x3 = x2 + step
        f3 = f(x3)

        if f3 > f2:
            break
        else:
            x1 = x2
            x2 = x3
            f1 = f2
            f2 = f3

        directional_df = pd.concat(
            [directional_df, pd.DataFrame([[x1, f1, x2, f2, x3, f3, complexity_rate]],
                                          columns=directional_df.columns)], ignore_index=True)

    # Комментарий не Михеева: здесь переводим числа в столбцах в шестнадцатиричную систему и делаем индексацию
    # строк с единицы, get_short_hex_output - не встроенная, привожу свою, можете написать сами
    # Конец комментария не Михеева
    columns_to_hex = ['x1', 'f1', 'x2', 'f2', 'x3', 'f3']
    directional_df[columns_to_hex] = directional_df[columns_to_hex].map(get_short_hex_output)
    directional_df.index = directional_df.index + 1

    # Печать таблицы о поиске триады
    print(directional_df)

    # directional_df_thriade - таблица, соответствующая работе с триадами
    directional_df_thriade = pd.DataFrame(
        columns=['x1', 'f1', 'x2', 'f2',
                 'x3', 'f3', 'Шаг'])

    # Этот флаг принимает значение 0, когда на предыдущей итерации знак шага поменялся на противоположный
    # (в этом случае он не делится на 2)
    change = True

    print('\n2) Работа с триадами')
    # Цикл работы с триадами
    while abs(step) >= 8 * 16 ** (- 4) * 2 ** n:
        if change:
            step = step / 2

        x_temp = x2 + step
        f_temp = f(x_temp)

        if f2 > f_temp:
            x1 = x2
            f1 = f2
            x2 = x_temp
            f2 = f_temp
            change = True
        else:
            if change:
                x3 = x1
                f3 = f1
                x1 = x_temp
                f1 = f_temp

                step = -step
                change = False
            else:
                x3 = x_temp
                f3 = f_temp

                change = True

        directional_df_thriade = pd.concat(
            [directional_df_thriade, pd.DataFrame([[x1, f1, x2, f2, x3, f3, step]],
                                                  columns=directional_df_thriade.columns)], ignore_index=True)

    # Комментарий не Михеева: здесь переводим числа в столбцах в шестнадцатиричную систему и делаем индексацию
    # строк с единицы
    # Конец комментария не Михеева
    columns_to_hex = ['x1', 'f1', 'x2', 'f2', 'x3', 'f3']
    directional_df_thriade[columns_to_hex] = directional_df_thriade[columns_to_hex].map(get_short_hex_output)
    directional_df_thriade.index = directional_df_thriade.index + 1

    # Печать таблицы о работе с триадами
    print(directional_df_thriade.to_string(col_space=15))

    return x2, f2
# Конец того, что Михеев сказал скинуть


di, f_di = directional_search(f, -10, 0.5)
print("Значение минимайзера: ", get_short_hex_output(di))
print("Значение функции: ", get_short_hex_output(f_di))