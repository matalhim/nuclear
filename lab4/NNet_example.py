import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class Network(object):

    def __init__(self, sizes):  # Инициализация сети с заданной архитектурой
        """
        Конструктор класса Network. Определяет количество слоев и их размеры, а также инициализирует веса и смещения случайными значениями.

        :param sizes: Список размеров слоев, где каждый элемент соответствует количеству нейронов в соответствующем слое.
        """
        self.num_layers = len(
            sizes)  # Количество слоев в сети (включая входной слой)
        self.sizes = sizes  # Сохранение списка размеров слоев
        self.biases = [np.random.randn(y, 1) for y in
                       # Инициализация смещений для каждого слоя, кроме первого (входного)
                       sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        # Инициализация весов между каждым слоем
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta, rand_num, test_data=None, pics=False):
        """
        Метод стохастического градиентного спуска (SGD). Обучает сеть на основе тренировочных данных.

        :param training_data: Набор обучающих данных
        :param epochs: Количество эпох обучения
        :param mini_batch_size: Размер мини-пакета
        :param eta: Скорость обучения
        :param test_data: Тестовые данные (опционально)
        """
        if test_data:  # Если тестовые данные переданы
            n_test = len(test_data)  # Количество примеров в тестовом наборе
        n = len(training_data)  # Количество примеров в обучающем наборе
        costs = []  # Список для хранения значений стоимости на каждой эпохе
        for j in range(epochs):  # Цикл по всем эпохам
            # Перемешиваем обучающие данные перед каждой эпохой
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            # Разделение данных на мини-пакеты
                            range(0, n, mini_batch_size)]
            # print(mini_batches)
            for mini_batch in mini_batches:  # Цикл по каждому мини-пакету
                # Обновление параметров сети на основе текущего мини-пакета
                self.update_mini_batch(mini_batch, eta)
            if test_data:  # Если есть тестовые данные
                pass
                # cost = self.total_cost(test_data)  # Вычисление общей стоимости на тестовой выборке
                # costs.append(cost)
                # print(f"Epoch {j}: Cost on test data: {cost}")
                # self.evaluate((test_data), n_test)  # Оценка точности модели на тестовых данных
            else:
                print("Epoch {0} complete".format(j))
            cost = self.total_cost(training_data)
            costs.append(cost)
            # print(f"Epoch {j}: Cost on test data: {cost}")
        if pics:
            plt.figure()
            plt.plot(range(epochs), costs, label='Cost')
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.title(f'Learning Rate: {eta}, Layers: {self.sizes}')
            plt.legend()
            plt.grid()
            filename = f'cost_plot_lr_{eta}_lay_{
                self.sizes}_ran_{rand_num}.png'
            plt.savefig(filename)
            plt.close()
            print(f'Saved plot as {filename}')

    def total_cost(self, test_data):
        """ Вычисление общей стоимости на тестовой выборке. """
        cost = 0
        for x, y in test_data:
            a = self.feedforward(x)
            cost += np.linalg.norm(a - y) ** 2  # Квадратичная ошибка
        return cost / len(test_data)

    def update_mini_batch(self, mini_batch, eta):
        """
        Обновляет параметры сети (весы и смещения) на основе одного мини-пакета.

        :param mini_batch: Мини-пакет данных
        :param eta: Скорость обучения
        """
        nabla_b = [np.zeros(
            b.shape) for b in self.biases]  # Инициализация накопителей для градиентов смещений
        # Инициализация накопителей для градиентов весов
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:  # Цикл по каждому примеру в мини-пакете
            delta_nabla_b, delta_nabla_w = self.backprop(
                x, y)  # Вычисляем градиенты для данного примера
            # Накапливаем градиенты смещений
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # Накапливаем градиенты весов
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw  # Обновляем веса
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb  # Обновляем смещения
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Реализация алгоритма обратного распространения ошибки (backpropagation).

        :param x: Входные данные
        :param y: Ожидаемые выходные данные
        :return: Градиенты смещений и весов
        """
        nabla_b = [np.zeros(
            b.shape) for b in self.biases]  # Инициализация накопителей для градиентов смещений
        # Инициализация накопителей для градиентов весов
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Прямой проход (feedforward)
        activation = x  # Начальная активация равна входному вектору
        activations = [x]  # Список всех активаций, начиная с входа
        zs = []  # Список всех значений Z (до применения активации)
        for b, w in zip(self.biases, self.weights):  # Цикл по всем слоям
            z = np.dot(w,
                       # Вычисление линейной комбинации весов и активации предыдущего слоя со смещением
                       activation) + b
            zs.append(z)  # Добавляем значение Z в список
            activation = self.sigmoid(z)  # Применение функции активации к Z
            # Добавляем результат активации в список
            activations.append(activation)
        # Обратное распространение ошибки (backward pass)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(
            zs[-1])  # Вычисление начальной дельты для последнего слоя
        # Обновление градиента смещения для последнего слоя
        nabla_b[-1] = delta
        # Обновление градиента весов для последнего слоя
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Цикл по остальным слоям (кроме выходного и первого скрытого)
        for l in range(2, self.num_layers):
            z = zs[-l]  # Получаем значение Z для текущего слоя
            # Производная сигмоиды для текущего слоя
            sp = self.sigmoid_prime(z)
            # Вычисление дельта для текущего слоя
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            # Обновление градиента смещения для текущего слоя
            nabla_b[-l] = delta
            # Обновление градиента весов для текущего слоя
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)  # Возвращаем градиенты смещений и весов

    def evaluate(self, test_data):
        """
        Оценивает точность модели на тестовых данных.

        :param test_data: Тестовый набор данных
        :return: Коэффициент детерминации R^2
        """
        test_results = []  # Список предсказанных значений
        real_values = []  # Список реальных значений
        for t, r in test_data:  # Цикл по каждому примеру в тестовом наборе
            ans = self.feedforward(t)[0][0]  # Прогнозируем значение
            # Добавляем прогнозированное значение в список
            test_results.append(ans)
            real_values.append(r[0][0])  # Добавляем реальное значение в список
        # Возвращаем коэффициент детерминации
        return abs(r2_score(test_results, real_values))

    def cost_derivative(self, output_activations, y):
        """
        Функция производной от функции стоимости.

        :param output_activations: Выходные активации последней нейросети
        :param y: Ожидаемый выходной сигнал
        :return: Производную функции стоимости
        """
        return (output_activations - y)  # Простая квадратичная функция стоимости

    def sigmoid(self, z):
        """
        Сигмоидная функция активации.

        :param z: Входное значение
        :return: Значение сигмоидной функции
        """
        return 1.0 / (1.0 + np.exp(-z))  # Стандартная формула сигмоида

    def feedforward(self, a):
        """
        Прямой проход через сеть. Возвращает выходное значение при подаче входного вектора `a`.

        :param a: Входной вектор
        :return: Выходное значение сети
        """
        for b, w in zip(self.biases, self.weights):  # Цикл по всем слоям
            # Вычисление линейной комбинации весов и активации предыдущего слоя со смещением
            z = np.dot(w, a) + b
            # Применение функции активации к Z
            a = self.sigmoid(np.dot(w, a) + b)
        return a  # Возвращаем итоговое выходное значение

    def sigmoid_prime(self, z):
        """
        Производная сигмоидной функции.

        :param z: Входное значение
        :return: Значение производной сигмоидальной функции
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))  # Формула производной сигмоиды


if __name__ == "__main__":
    a = pd.read_csv("Cu_conc_in_spectrum.txt",
                    delimiter=' ', names=['y', 'names'])
    Cu_Ka = [0, 0]
    Cu_Kb = [0, 0]
    with open("elements_windows.txt") as file:
        for line in file:
            if line[:5] == "Cu_Ka":
                Cu_Ka[0] = int(line[6:9])
                Cu_Ka[1] = int(line[10:13])
            elif line[:5] == "Cu_Kb":
                Cu_Kb[0] = int(line[6:9])
                Cu_Kb[1] = int(line[10:13])

    x1 = []
    x2 = []
    for name in a['names']:
        with open(fr'lab4/spectrums/{name}.mca', encoding='unicode-escape') as file:
            counter = 1
            times = 0
            summa = 0
            summb = 0
            for line in file:
                if line[:3] == "Acc":
                    times = float(line[19:])
                if Cu_Ka[0] <= counter <= Cu_Ka[1]:
                    summa += int(line)
                if Cu_Kb[0] <= counter <= Cu_Kb[1]:
                    summb += int(line)
                counter += 1
            x1.append(summa / times)
            x2.append(summb / times)

    y = np.array(a['y'].tolist())
    data = []
    for i in range(len(x1)):
        data.append(
            (np.array([[x1[i] / max(x1)], [x2[i] / max(x2)]]), np.array([[y[i] / max(y)]])))
    min_ans = []
    file = open("lab4/iwannadie.txt", 'w')
    for le in [2, 3, 4]:
        for eta in [0.1, 0.5, 1]:
            for epo in [300, 400, 500, 600]:
                for i in range(100):
                    start_time = time.time()
                    training_data, test_data = train_test_split(
                        data, test_size=0.3, random_state=i)

                    layers = [2, le, 1]
                    net = Network(layers)
                    net.SGD(training_data, epo, len(training_data),
                            eta, i, test_data=test_data)

                    result = []
                    true_data = []
                    for entry in test_data:
                        received_value = net.feedforward(entry[0])
                        # print(" Expected: ", entry[1], " Received : ", received_value)
                        result.append(received_value[0][0])
                        true_data.append(entry[1][0][0])
                    # print(true_data)
                    # print(result)
                    # plt.figure()
                    # plt.scatter(true_data, result)
                    # plt.xlabel('Test Data')
                    # plt.ylabel('Predicted Data')
                    # plt.title(f'Learning Rate: {eta}, Layers: {layers}')
                    # plt.grid()
                    # # plt.yscale("log")
                    # filename = f'compare_data_ran_lr_{eta}_lay_{layers}_ran_{i}.png'
                    # plt.savefig(filename)
                    # plt.close()
                    # print(f'Saved plot as {filename}')

                    ans = net.evaluate(test_data)  # r2 score calculation
                    end_time = time.time()

                    print(f"le {le}, eta {eta}, epo {epo}, i {i}-> {ans}")
                    file.write(f"{ans}\t{i}\t{layers}\t{eta}\t{
                               epo}\t{end_time - start_time}\n")
                    min_ans.append(
                        [ans, i, layers, eta, epo, end_time - start_time])
    min_ans = sorted(min_ans, key=lambda m_ans: m_ans[0])
    file.close()

    for i in range(6):
        print(f"{i + 1} place by ans: {round(min_ans[i][0], 4)} with:\tRandom_state = {min_ans[i][1]},\tLayers = {min_ans[i][2]},\t"
              f"Learning rate = {min_ans[i][3]},\tEpochs = {min_ans[i][4]},\tTime = {min_ans[i][5]} s")
        training_data, test_data = train_test_split(
            data, test_size=0.3, random_state=min_ans[i][1])

        layers = min_ans[i][2]
        net = Network(layers)
        net.SGD(training_data, min_ans[i][4], len(
            training_data), min_ans[i][3], min_ans[i][1], test_data=test_data, pics=True)

        result = []
        true_data = []
        for entry in test_data:
            received_value = net.feedforward(entry[0])
            # print(" Expected: ", entry[1], " Received : ", received_value)
            result.append(received_value[0][0])
            true_data.append(entry[1][0][0])
        ans = net.evaluate(test_data)

        plt.figure()
        plt.scatter(true_data, result, label=f"{round(ans, 4)}")
        plt.xlabel('Test Data')
        plt.ylabel('Predicted Data')
        plt.title(f'Learning Rate: {min_ans[i][3]}, Layers: {layers}')
        plt.grid()
        plt.legend()
        filename = f'compare_data_ran_lr_{min_ans[i][3]}_lay_{
            layers}_ran_{min_ans[i][1]}.png'
        plt.savefig(filename)
        plt.close()
        print(f'Saved plot as {filename}')
