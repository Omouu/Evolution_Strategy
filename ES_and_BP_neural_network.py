'''
进化策略神经网络对比代码
BP, (mu+lambda)-ES, (mu,lambda)-ES。
评估函数：y = 1 / (x^2 + 1)
'''
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


class BPNN(object):
    '梯度下降BP神经网络。'
    def __init__(self, shape, eta, data, epochs):
        '''网络参数随机初始化。
        shape:网络的形状。由元组表示，长度为总层数，每个元素代表该层的神经元数量。
        eta:学习速率。
        '''
        self.shape = shape
        self.eta = eta
        self.layer_num = len(self.shape)
        self.epoch = 0
        self.data = data
        self.epochs = epochs

        # 初始化每两层节点之间的权重矩阵。行数为高层神经元数，列数为低层神经元数。
        self.weights = [
            np.matrix(np.random.randn(m, n))
            for m, n in zip(self.shape[1:], self.shape[:-1])
        ]
        # 阈值矩阵。每层网络的矩阵行数为1，列数为神经元数。
        self.biases = [
            np.matrix(np.random.randn(1, n)) for n in self.shape[1:]
        ]

    def feedforward(self, X):
        '''前向计算样本的结果。
            X: 样本输入
            Y: 网络的输出
        '''
        for i in range(self.layer_num - 1):
            X = self.sigmoid(X * self.weights[i].T + self.biases[i])
        return X

    def backpropagation(self, X, Y):
        '''单个样本权重和阈值的更新。
        Args：
            X：样本输入
            Y：样本label
        '''
        # 计算每个神经元激活前和激活后的值。
        # 除输入层外每层未经激活的神经元。
        inactive = []
        # 每层经过激活的神经元（输入层除外）。
        active = [X]

        for i in range(self.layer_num - 1):
            inactive.append((active[i] * self.weights[i].T +
                            self.biases[i]).A)
            active.append(self.sigmoid(inactive[i]))

        # 计算每个神经元的梯度项。
        # 梯度项矩阵初始化, 元素为array对象。
        delta = [None] * (self.layer_num - 1)
        # 输出层的梯度项。
        delta[-1] = active[-1] * (1 - active[-1]) * (active[-1] - Y)
        # 倒层序计算其他层的梯度项。
        for i in range(2, self.layer_num):
            delta[-i] = delta[-i + 1] * self.weights[-i + 1]
            delta[-i] = (active[-i] * (1 - active[-i])) * delta[-i].A

        # 更新权重和阈值矩阵。
        for i in range(self.layer_num - 1):
            self.weights[i] -= np.matrix(delta[i].T) * active[i] * self.eta
            self.biases[i] -= -delta[i] * self.eta

    def sigmoid(self, X):
        '激活函数: 对率函数Sigmoid。'
        return 1 / (1 + np.exp(-X))

    def evaluate(self, test_set):
        '评估网络。'
        E = [self.feedforward(x).A - y for (x, y) in test_set]
        E_mean = (np.array(E) ** 2).mean()
        print('Epoch %d: E_mean %f' % (self.epoch, E_mean))
        return E_mean

    def training(self):
        '训练网络。'
        errors = []

        for i in range(self.epochs):
            self.epoch += 1
            random.shuffle(self.data[0])
            for x, y in self.data[0]:
                self.backpropagation(x, y)
            errors.append(self.evaluate((self.data[1])))

        print('Training complete.')
        return errors


class NN(object):
    def __init__(self, shape, init_flag):
        '''
        shape:网络的形状。由元组表示，长度为总层数，每个元素代表该层的神经元数量。
        eta:学习速率。
        '''
        self.shape = shape
        self.layer_num = len(self.shape)

        if init_flag:
            '常规初始化。'
            # 初始化每两层节点之间的权重矩阵。行数为高层神经元数，列数为低层神经元数。
            self.weights = [
                np.random.randn(m, n)
                for m, n in zip(self.shape[1:], self.shape[:-1])
            ]
            self.weights_mute_strength = [
                np.random.rand(m, n)
                for m, n in zip(self.shape[1:], self.shape[:-1])
            ]

            # 阈值矩阵。每层网络的矩阵行数为1，列数为神经元数。
            self.biases = [np.random.randn(1, n) for n in self.shape[1:]]
            self.biases_mute_strength = [
                np.random.rand(1, n) for n in self.shape[1:]
            ]
        else:
            '初始化网络容器。'
            self.weights = [
                np.array(np.empty((m, n)))
                for m, n in zip(self.shape[1:], self.shape[:-1])
            ]
            self.weights_mute_strength = self.weights
            self.biases = [
                np.array(np.empty((1, n))) for n in self.shape[1:]
            ]
            self.biases_mute_strength = self.biases

    def feedforward(self, X):
        '前向计算网络输出。'
        for i in range(self.layer_num - 1):
            X = self.sigmoid(X * np.matrix(self.weights[i]).T +
                             np.matrix(self.biases[i]))
        return X

    def sigmoid(self, X):
        '激活函数: 对率函数Sigmoid。'
        return 1 / (1 + np.exp(-X))


class ESPNN(object):
    '(mu+lambda)-ES'
    def __init__(self, shape, population_size, filial_size, generation, data):
        '种群初始化。'
        self.nn_shape = shape
        self.nn_layer_num = len(self.nn_shape)
        self.pop_size = population_size
        self.population = [
            NN(shape, True) for _ in range(self.pop_size)
        ]
        self.filial_size = filial_size
        self.filial = []
        self.probability = self.GD_pd()
        self.data_set = data
        self.generation = generation
        self.generation_record = 0

    def crossover(self, fitting_data):
        # 创建子代容器
        self.filial = [
            NN(self.nn_shape, True) for i in range(self.filial_size)]

        # 父代个体按适应度排序，优秀父代在前
        fitnesses = [
            self.fitness(indv, fitting_data) for indv in self.population]
        p_sorted = np.array(fitnesses).argsort()[::-1]
        for kid in self.filial:
            # 随机选择两个父代
            p1, p2 = np.random.choice(p_sorted, size=2, p=self.probability)

            for i in range(self.nn_layer_num - 1):
                # 权重矩阵交叉点（cross point）
                cp_w = np.random.choice([True, False],
                                        size=kid.weights[i].shape)
                # 交叉权重矩阵和权重突变矩阵
                kid.weights[i][cp_w] = self.population[p1].weights[i][cp_w]
                kid.weights[i][~cp_w] = self.population[p2].weights[i][~cp_w]
                kid.weights_mute_strength[i][cp_w] = \
                    self.population[p1].weights_mute_strength[i][cp_w]
                kid.weights_mute_strength[i][~cp_w] = \
                    self.population[p1].weights_mute_strength[i][~cp_w]

                # 阈值矩阵交叉点
                cp_b = np.random.choice([True, False],
                                        size=kid.biases[i].shape)
                # 交叉阈值矩阵
                kid.biases[i][cp_b] = self.population[p1].biases[i][cp_b]
                kid.biases[i][~cp_b] = self.population[p2].biases[i][~cp_b]
                kid.biases_mute_strength[i][cp_b] = \
                    self.population[p1].biases_mute_strength[i][cp_b]
                kid.biases_mute_strength[i][~cp_b] = \
                    self.population[p2].biases_mute_strength[i][~cp_b]

    def mutation(self):
        for kid in self.filial:
            for i in range(self.nn_layer_num - 1):
                # 突变强度的突变，保持物种多样性
                w_shape = kid.weights_mute_strength[i].shape
                kid.weights_mute_strength[i] += np.random.rand(*w_shape)-0.5
                kid.weights_mute_strength[i] = \
                    np.maximum(kid.weights_mute_strength[i] * 5, 0)

                b_shape = kid.biases_mute_strength[i].shape
                kid.biases_mute_strength[i] += np.random.rand(*b_shape)-0.5
                kid.biases_mute_strength[i] = \
                    np.maximum(kid.biases_mute_strength[i] * 5, 0)

                # 突变
                kid.weights[i] += \
                    kid.weights_mute_strength[i] * np.random.randn(*w_shape)
                kid.biases[i] += \
                    kid.biases_mute_strength[i] * np.random.randn(*b_shape)

    def selection(self, fitting_data):
        pop = self.population + self.filial
        fitnesses = [self.fitness(indv, fitting_data) for indv in pop]
        good_idx = np.array(fitnesses).argsort()[-self.pop_size:]
        self.population = [pop[i] for i in good_idx]

        print("mean: %.4f  max: %.4f" % (sum(fitnesses) / len(fitnesses),
              max(fitnesses)), end='...')

    def fitness(self, nn, fitting_data):
        fitnesses = 0
        for x, y in fitting_data:
            Y = nn.feedforward(x)
            E = ((Y - y).A ** 2).mean()
            fitnesses += E
        return 1 * len(fitting_data) / fitnesses

    def GD_pd(self):
        'Gaussian Distribution probability density function.'
        x = np.linspace(0, 2, self.pop_size)
        distribution = 0.4 * np.exp(-0.5 * (x ** 2))
        distribution /= distribution.sum()
        distribution[0] -= abs(1 - distribution.sum())
        return distribution

    def evaluate(self, x, y):
        '评估种群误差。'
        error = []
        for i in self.population:
            r = [i.feedforward(np.matrix(inp)).A[0][0] for inp in x]
            error.append(((np.array(r) - y) ** 2).mean())
        return np.array(error)

    def training(self):
        print("Training start.")

        # 评估用数据
        x = np.linspace(-5, 5, 100)
        y = 1 / (1 + x ** 2)

        error = []
        for g in range(self.generation):
            print("Generation: %d" % self.generation_record, end='...')
            self.generation_record += 1
            print("crossover", end='...')
            self.crossover(self.data_set[1])
            print("mutation", end='...')
            self.mutation()
            print("selection", end='...')
            self.selection(self.data_set[0])
            print("evaluate...")
            error.append(self.evaluate(x, y))

        return error

    def fitting(self, x):
        fitnesses = [
            self.fitness(indv, self.data_set[0]) for indv in self.population]
        best_idx = np.array(fitnesses).argsort()[-1]
        r = [
            self.population[best_idx].feedforward(np.matrix(inp)).A[0][0]
            for inp in x]
        return r


class ESNN(object):
    '(mu,lambda)-ES'
    def __init__(self, shape, population_size, filial_size, generation, data):
        '种群初始化。'
        self.nn_shape = shape
        self.nn_layer_num = len(self.nn_shape)
        self.pop_size = population_size
        self.population = [
            NN(shape, True) for _ in range(self.pop_size)
        ]
        self.filial_size = filial_size
        self.filial = []
        self.f_probability = self.GD_pd(self.filial_size - 10)
        self.p_probability = self.GD_pd(self.pop_size)
        self.data_set = data
        self.generation = generation
        self.generation_record = 0

    def crossover(self, fitting_data):
        # 创建子代容器
        self.filial = [
            NN(self.nn_shape, True) for i in range(self.filial_size)]

        # 父代个体按适应度排序，优秀父代在前
        fitnesses = [
            self.fitness(indv, fitting_data) for indv in self.population]
        p_sorted = np.array(fitnesses).argsort()[::-1]
        for kid in self.filial:
            # 随机选择两个父代
            p1, p2 = np.random.choice(p_sorted, size=2, p=self.p_probability)

            for i in range(self.nn_layer_num - 1):
                # 权重矩阵交叉点（cross point）
                cp_w = np.random.choice([True, False],
                                        size=kid.weights[i].shape)
                # 交叉权重矩阵和权重突变矩阵
                kid.weights[i][cp_w] = self.population[p1].weights[i][cp_w]
                kid.weights[i][~cp_w] = self.population[p2].weights[i][~cp_w]
                kid.weights_mute_strength[i][cp_w] = \
                    self.population[p1].weights_mute_strength[i][cp_w]
                kid.weights_mute_strength[i][~cp_w] = \
                    self.population[p1].weights_mute_strength[i][~cp_w]

                # 阈值矩阵交叉点
                cp_b = np.random.choice([True, False],
                                        size=kid.biases[i].shape)
                # 交叉阈值矩阵
                kid.biases[i][cp_b] = self.population[p1].biases[i][cp_b]
                kid.biases[i][~cp_b] = self.population[p2].biases[i][~cp_b]
                kid.biases_mute_strength[i][cp_b] = \
                    self.population[p1].biases_mute_strength[i][cp_b]
                kid.biases_mute_strength[i][~cp_b] = \
                    self.population[p2].biases_mute_strength[i][~cp_b]

    def mutation(self):
        for kid in self.filial:
            for i in range(self.nn_layer_num - 1):
                # 突变强度的突变，保持物种多样性
                w_shape = kid.weights_mute_strength[i].shape
                kid.weights_mute_strength[i] += np.random.rand(*w_shape)-0.5
                kid.weights_mute_strength[i] = \
                    np.maximum(kid.weights_mute_strength[i], 0)

                b_shape = kid.biases_mute_strength[i].shape
                kid.biases_mute_strength[i] += np.random.rand(*b_shape)-0.5
                kid.biases_mute_strength[i] = \
                    np.maximum(kid.biases_mute_strength[i], 0)

                # 突变
                kid.weights[i] += \
                    kid.weights_mute_strength[i] * np.random.randn(*w_shape)
                kid.biases[i] += \
                    kid.biases_mute_strength[i] * np.random.randn(*b_shape)

    def selection(self, fitting_data):
        fitnesses = [self.fitness(indv, fitting_data) for indv in self.filial]
        id_sorted = np.array(fitnesses).argsort()[::-1]
        # 保留排名前十的个体
        self.population = [self.filial[i] for i in id_sorted[:10]]
        good_idx = np.random.choice(id_sorted[10:],
                                    size=(self.pop_size - 10),
                                    p=self.f_probability)
        self.population += [self.filial[i] for i in good_idx]

        print("mean: %.4f  max: %.4f" % (sum(fitnesses) / len(fitnesses),
              max(fitnesses)), end='...')

    def fitness(self, nn, fitting_data):
        fitnesses = 0
        for x, y in fitting_data:
            Y = nn.feedforward(x)
            E = ((Y - y).A ** 2).mean()
            fitnesses += E
        return 1 * len(fitting_data) / fitnesses

    def GD_pd(self, x):
        'Gaussian Distribution probability density function.'
        x = np.linspace(0, 2, x)
        distribution = 0.4 * np.exp(-0.5 * (x ** 2))
        distribution /= distribution.sum()
        distribution[0] -= abs(1 - distribution.sum())
        return distribution

    def evaluate(self, x, y):
        '评估种群误差。'
        error = []
        for i in self.population:
            r = [i.feedforward(np.matrix(inp)).A[0][0] for inp in x]
            error.append(((np.array(r) - y) ** 2).mean())
        return np.array(error)

    def training(self):
        print("Training start.")

        # 评估用数据
        x = np.linspace(-5, 5, 100)
        y = 1 / (1 + x ** 2)

        error = []
        for g in range(self.generation):
            print("Generation: %d" % self.generation_record, end='...')
            self.generation_record += 1
            print("crossover", end='...')
            self.crossover(self.data_set[1])
            print("mutation", end='...')
            self.mutation()
            print("selection", end='...')
            self.selection(self.data_set[0])
            print("evaluate...")
            error.append(self.evaluate(x, y))

        return error

    def fitting(self, x):
        fitnesses = [
            self.fitness(indv, self.data_set[0]) for indv in self.population]
        best_idx = np.array(fitnesses).argsort()[-1]
        r = [
            self.population[best_idx].feedforward(np.matrix(inp)).A[0][0]
            for inp in x]
        return r


def one():
    '''
    隐层节点数10 四种网络的曲线拟合误差随代数的变化的变化
    100代 三种网络的最终曲线拟合误差随隐层节点数的变化的变化
    '''
    # 数据预处理
    # 拟合曲线y=1/(x^2+1),输入范围[-5,5],输出范围(0,1],采样1000
    x = np.linspace(-5, 5, 1000)
    data = [[np.matrix(i), np.matrix(1 / (i ** 2 + 1))] for i in x]
    data = [data, random.sample(data, 500)]

    # 三种网络的曲线拟合结果比较 plot图
    bpnn = BPNN(shape=[1, 6, 1],
                eta=0.3,
                data=data,
                epochs=100)
    bpnn_error_per_e = bpnn.training()

    espnn = ESPNN(shape=[1, 6, 1],
                  population_size=100,
                  filial_size=20,
                  generation=100,
                  data=data)
    espnn_error_per_g = espnn.training()

    esnn = ESNN(shape=[1, 6, 1],
                population_size=50,
                filial_size=200,
                generation=100,
                data=data)
    esnn_error_per_g = esnn.training()

    np.savez('1.npz',
             bpnn_error_per_e=bpnn_error_per_e,
             espnn_error_per_g=espnn_error_per_g,
             esnn_error_per_g=esnn_error_per_g)

    # 绘制拟合曲线图
    plt.figure(num=1)
    # 原函数曲线
    lo, = plt.plot(x, 1 / (x ** 2 + 1))
    # 拟合曲线
    lbp, = plt.plot(x, [bpnn.feedforward(np.matrix(i)).A[0][0] for i in x])
    lesp, = plt.plot(x, espnn.fitting(x))
    les, = plt.plot(x, esnn.fitting(x))
    plt.legend(handles=[lo, lbp, lesp, les],
               labels=['Origin Curve',
                       'BP Fitting Curve',
                       u'(μ+λ)-ES Fitting Curve',
                       u'(μ,λ)-ES Fitting Curve'],
               loc='best')

    plt.figure(num=2)
    bp_m, = plt.plot(np.arange(100), 1 / np.array(bpnn_error_per_e))
    esp_mean, = plt.plot(np.arange(100),
                         [1 / i.mean() for i in espnn_error_per_g])
    esp_min, = plt.plot(np.arange(100),
                        [1 / i.min() for i in espnn_error_per_g])
    es_mean, = plt.plot(np.arange(100),
                        [1 / i.mean() for i in esnn_error_per_g])
    es_min, = plt.plot(np.arange(100),
                       [1 / i.min() for i in esnn_error_per_g])
    plt.legend(handles=[bp_m, esp_mean, es_mean, esp_min, es_min],
               labels=['BP',
                       u'mean: (μ+λ)-ES',
                       u'mean: (μ,λ)-ES',
                       u'min: (μ+λ)-ES',
                       u'min: (μ,λ)-ES'],
               loc='best')

    plt.show()


def two():
    'BP神经网络'
    x = np.linspace(-5, 5, 1000)
    data = [[np.matrix(i), np.matrix(1 / (i ** 2 + 1))] for i in x]
    data = [data, random.sample(data, 500)]

    xx = np.arange(5, 200, 5)

    jet = cm = plt.get_cmap('jet') 
    cNorm = colors.Normalize(vmin=0, vmax=(len(xx) - 1))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap='Blues')

    for i in range(len(xx)):
        bpnn = BPNN(shape=[1, xx[i], 1],
                    eta=0.3,
                    data=data,
                    epochs=100)
        plt.plot(np.arange(100),
                 1 / np.array(bpnn.training()),
                 color=scalarMap.to_rgba(i))
        print(xx[i])

    plt.xlabel('echos')
    plt.ylabel('fitness')
    plt.show()


# one()
two()
# mu,lambda 曲线拟合误差随种群大小的变化的变化 plot图
