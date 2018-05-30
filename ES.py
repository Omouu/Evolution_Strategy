'''
(mu+lambda)-ES
(mu,lambda)-ES
(mu+lambda)-EP
(mu,lambda)-EP
'''
import numpy as np
import matplotlib.pyplot as plt


class ES_P():
    def __init__(self, DNA_SIZE, DNA_BOUND, GENERATIONS, POP_SIZE, OFF_SIZE):
        self.DNA_SIZE = DNA_SIZE
        self.DNA_BOUND = DNA_BOUND
        self.GENERATIONS = GENERATIONS
        self.POP_SIZE = POP_SIZE
        self.OFFSPRING_SIZE = OFF_SIZE

        self.pop = dict(
            DNA=(np.random.rand(1, self.DNA_SIZE) - 0.5
                 ).repeat(self.POP_SIZE, axis=0),
            mut_strength=np.random.rand(self.POP_SIZE, self.DNA_SIZE)
        )
        self.probability = self.GD_pd()

    def GD_pd(self):
        'Gaussian Distribution probability density function.'
        x = np.linspace(0, 2, self.POP_SIZE)
        distribution = 0.4 * np.exp(-0.5 * (x ** 2))
        distribution /= distribution.sum()
        distribution[0] -= abs(1 - distribution.sum())
        return distribution

    def crossover_and_mutation(self, pop, n_kid):
        fitness = np.array([rastrigin(*i) for i in self.pop['DNA']])
        good_idx = fitness.argsort()[::-1]

        # generate empty kid holder
        kids = {'DNA': np.empty((n_kid, self.DNA_SIZE))}
        kids['mut_strength'] = np.empty_like(kids['DNA'])
        for dna, mute in zip(kids['DNA'], kids['mut_strength']):
            # crossover (roughly half p1 and half p2)
            p1, p2 = np.random.choice(good_idx, size=2, p=self.probability)
            # crossover points
            cp = np.random.randint(0, 2, self.DNA_SIZE, dtype=np.bool)
            dna[cp] = pop['DNA'][p1, cp]
            dna[~cp] = pop['DNA'][p2, ~cp]
            mute[cp] = pop['mut_strength'][p1, cp]
            mute[~cp] = pop['mut_strength'][p2, ~cp]

            # mutate (change DNA based on normal distribution)
            mute[:] = np.maximum(mute + (np.random.rand(*mute.shape)-0.5), 0.)
            dna += mute * np.random.randn(*dna.shape)
            dna[:] = np.clip(dna, *self.DNA_BOUND)
        return kids

    def selection(self, pop, kids):
        # put pop and kids together
        for key in ['DNA', 'mut_strength']:
            pop[key] = np.vstack((pop[key], kids[key]))
        # calculate global fitness
        fitness = np.array([rastrigin(*i) for i in pop['DNA']])
        idx = np.arange(pop['DNA'].shape[0])
        # selected by fitness ranking (not value)
        good_idx = idx[fitness.argsort()][-self.POP_SIZE:]
        for key in ['DNA', 'mut_strength']:
            pop[key] = pop[key][good_idx]
        return pop

    def training(self):
        pop_per_generation = []
        for _ in range(self.GENERATIONS):
            pop_per_generation.append(self.pop['DNA'])
            kids = self.crossover_and_mutation(self.pop, self.OFFSPRING_SIZE)
            pop = self.selection(self.pop, kids)
        return pop_per_generation


class ES_():
    def __init__(self, DNA_SIZE, DNA_BOUND, GENERATIONS, POP_SIZE, OFF_SIZE):
        self.DNA_SIZE = DNA_SIZE
        self.DNA_BOUND = DNA_BOUND
        self.GENERATIONS = GENERATIONS
        self.POP_SIZE = POP_SIZE
        self.OFFSPRING_SIZE = OFF_SIZE

        self.pop = dict(
            DNA=(np.random.rand(1, self.DNA_SIZE) - 0.5
                 ).repeat(self.POP_SIZE, axis=0),
            mut_strength=np.random.rand(self.POP_SIZE, self.DNA_SIZE)
        )
        self.probability = self.GD_pd()

    def GD_pd(self):
        'Gaussian Distribution probability density function.'
        x = np.linspace(0, 2, self.POP_SIZE)
        distribution = 0.4 * np.exp(-0.5 * (x ** 2))
        distribution /= distribution.sum()
        distribution[0] -= abs(1 - distribution.sum())
        return distribution

    def crossover_and_mutation(self, pop, n_kid):
        fitness = np.array([rastrigin(*i) for i in self.pop['DNA']])
        good_idx = fitness.argsort()[::-1]

        # generate empty kid holder
        kids = {'DNA': np.empty((n_kid, self.DNA_SIZE))}
        kids['mut_strength'] = np.empty_like(kids['DNA'])
        for dna, mute in zip(kids['DNA'], kids['mut_strength']):
            # crossover (roughly half p1 and half p2)
            p1, p2 = np.random.choice(good_idx, size=2, p=self.probability)
            # crossover points
            cp = np.random.randint(0, 2, self.DNA_SIZE, dtype=np.bool)
            dna[cp] = pop['DNA'][p1, cp]
            dna[~cp] = pop['DNA'][p2, ~cp]
            mute[cp] = pop['mut_strength'][p1, cp]
            mute[~cp] = pop['mut_strength'][p2, ~cp]

            # mutate (change DNA based on normal distribution)
            mute[:] = np.maximum(mute + (np.random.rand(*mute.shape)-0.5), 0.)
            dna += mute * np.random.randn(*dna.shape)
            dna[:] = np.clip(dna, *self.DNA_BOUND)
        return kids

    def selection(self, kids):
        self.pop = kids
        # calculate global fitness
        fitness = np.array([rastrigin(*i) for i in self.pop['DNA']])
        idx = np.arange(self.pop['DNA'].shape[0])
        # selected by fitness ranking (not value)
        good_idx = idx[fitness.argsort()][-self.POP_SIZE:]
        for key in ['DNA', 'mut_strength']:
            self.pop[key] = self.pop[key][good_idx]

    def training(self):
        pop_per_generation = []
        for _ in range(self.GENERATIONS):
            pop_per_generation.append(self.pop['DNA'])
            kids = self.crossover_and_mutation(self.pop, self.OFFSPRING_SIZE)
            self.selection(kids)
        return pop_per_generation


class EP_P():
    def __init__(self, DNA_SIZE, DNA_BOUND, GENERATIONS, POP_SIZE, OFF_SIZE):
        self.DNA_SIZE = DNA_SIZE
        self.DNA_BOUND = DNA_BOUND
        self.GENERATIONS = GENERATIONS
        self.POP_SIZE = POP_SIZE
        self.OFFSPRING_SIZE = OFF_SIZE

        self.pop = dict(
            DNA=(np.random.rand(1, self.DNA_SIZE) - 0.5
                 ).repeat(self.POP_SIZE, axis=0),
            mut_strength=np.random.rand(self.POP_SIZE, self.DNA_SIZE)
        )
        self.probability = self.GD_pd()

    def GD_pd(self):
        'Gaussian Distribution probability density function.'
        x = np.linspace(0, 2, self.POP_SIZE)
        distribution = 0.4 * np.exp(-0.5 * (x ** 2))
        distribution /= distribution.sum()
        distribution[0] -= abs(1 - distribution.sum())
        return distribution

    def mutation(self, pop, n_kid):
        # 按随适应度递减的概率抽取子代
        fitness = np.array([rastrigin(*i) for i in self.pop['DNA']])
        good_idx = np.random.choice(fitness.argsort()[::-1],
                                    size=self.OFFSPRING_SIZE,
                                    p=self.probability)
        kids = {}
        for key in ['DNA', 'mut_strength']:
            kids[key] = self.pop[key][good_idx]

        for dna, mute in zip(kids['DNA'], kids['mut_strength']):
            # mutate (change DNA based on normal distribution)
            mute[:] = np.maximum(mute + (np.random.rand(*mute.shape)-0.5), 0.)
            dna += mute * np.random.randn(*dna.shape)
            dna[:] = np.clip(dna, *self.DNA_BOUND)
        return kids

    def selection(self, pop, kids):
        # put pop and kids together
        for key in ['DNA', 'mut_strength']:
            pop[key] = np.vstack((pop[key], kids[key]))
        # calculate global fitness
        fitness = np.array([rastrigin(*i) for i in pop['DNA']])
        idx = np.arange(pop['DNA'].shape[0])
        # selected by fitness ranking (not value)
        good_idx = idx[fitness.argsort()][-self.POP_SIZE:]
        for key in ['DNA', 'mut_strength']:
            pop[key] = pop[key][good_idx]
        return pop

    def training(self):
        pop_per_generation = []
        for _ in range(self.GENERATIONS):
            pop_per_generation.append(self.pop['DNA'])
            kids = self.mutation(self.pop, self.OFFSPRING_SIZE)
            pop = self.selection(self.pop, kids)
        return pop_per_generation


class EP_():
    def __init__(self, DNA_SIZE, DNA_BOUND, GENERATIONS, POP_SIZE, OFF_SIZE):
        self.DNA_SIZE = DNA_SIZE
        self.DNA_BOUND = DNA_BOUND
        self.GENERATIONS = GENERATIONS
        self.POP_SIZE = POP_SIZE
        self.OFFSPRING_SIZE = OFF_SIZE

        self.pop = dict(
            DNA=(np.random.rand(1, self.DNA_SIZE) - 0.5
                 ).repeat(self.POP_SIZE, axis=0),
            mut_strength=np.random.rand(self.POP_SIZE, self.DNA_SIZE)
        )
        self.probability = self.GD_pd()

    def GD_pd(self):
        'Gaussian Distribution probability density function.'
        x = np.linspace(0, 2, self.POP_SIZE)
        distribution = 0.4 * np.exp(-0.5 * (x ** 2))
        distribution /= distribution.sum()
        distribution[0] -= abs(1 - distribution.sum())
        return distribution

    def mutation(self, pop, n_kid):
        # 按随适应度递减的概率抽取子代
        fitness = np.array([rastrigin(*i) for i in self.pop['DNA']])
        good_idx = np.random.choice(fitness.argsort()[::-1],
                                    size=self.OFFSPRING_SIZE,
                                    p=self.probability)
        kids = {}
        for key in ['DNA', 'mut_strength']:
            kids[key] = self.pop[key][good_idx]

        for dna, mute in zip(kids['DNA'], kids['mut_strength']):
            # mutate (change DNA based on normal distribution)
            mute[:] = np.maximum(mute + (np.random.rand(*mute.shape)-0.5), 0.)
            dna += mute * np.random.randn(*dna.shape)
            dna[:] = np.clip(dna, *self.DNA_BOUND)
        return kids

    def selection(self, kids):
        self.pop = kids
        # calculate global fitness
        fitness = np.array([rastrigin(*i) for i in self.pop['DNA']])
        idx = np.arange(self.pop['DNA'].shape[0])
        # selected by fitness ranking (not value)
        good_idx = idx[fitness.argsort()][-self.POP_SIZE:]
        for key in ['DNA', 'mut_strength']:
            self.pop[key] = self.pop[key][good_idx]

    def training(self):
        pop_per_generation = []
        for _ in range(self.GENERATIONS):
            pop_per_generation.append(self.pop['DNA'])
            kids = self.mutation(self.pop, self.OFFSPRING_SIZE)
            self.selection(kids)
        return pop_per_generation


def rastrigin(x, y):
    r = 10 * 2 + x ** 2 + y ** 2 - 10 * np.cos(2 * np.pi * x) -\
        10 * np.cos(2 * np.pi * y)
    return r / 100


def drawing(data):
    # initialization of drawing
    x = np.linspace(-6, 6, 250)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(X, Y)
    ax = plt.subplot(111)
    # draw filled contour
    cf = ax.contourf(X, Y, Z, 8, alpha=0.8, cmap=plt.cm.hot)
    plt.colorbar(cf)

    plt.pause(15)
    for i in data:
        try:
            sca.remove()
        except UnboundLocalError:
            pass
        sca = ax.scatter(i.T[0], i.T[1], s=100, c='#441c94', alpha=0.7)
        # plt.savefig("C:\\Users\\Omou\\Desktop\\bs\\论文\\1\\%d.png" % _)
        plt.pause(0.1)

    plt.show()


def mean(x):
    result = np.zeros(60)
    for i in x:
        mean = []
        for g in i:
            m_per_g = [rastrigin(x, y) for x, y in g]
            mean.append(np.array(m_per_g).mean())
        result += np.array(mean)
    return result / 50


def max_(x):
    result = np.zeros(60)
    for i in x:
        max_ = []
        for g in i:
            m_per_g = [rastrigin(x, y) for x, y in g]
            max_.append(max(m_per_g))
        result += np.array(max_)
    return result / 50


def one():
    es_p_pops = []
    es__pops = []
    ep_p_pops = []
    ep__pops = []

    for i in range(50):
        # (mu+lambda)-ES
        es_p = ES_P(DNA_SIZE=2,
                    DNA_BOUND=[-5.5, 5.5],
                    GENERATIONS=60,
                    POP_SIZE=100,
                    OFF_SIZE=50)
        es_p_pops.append(es_p.training())

        # (mu,lambda)-ES
        es_ = ES_(DNA_SIZE=2,
                  DNA_BOUND=[-5.5, 5.5],
                  GENERATIONS=60,
                  POP_SIZE=100,
                  OFF_SIZE=200)
        es__pops.append(es_.training())

        # (mu+lambda)-EP
        ep_p = EP_P(DNA_SIZE=2,
                    DNA_BOUND=[-5.5, 5.5],
                    GENERATIONS=60,
                    POP_SIZE=100,
                    OFF_SIZE=50)
        ep_p_pops.append(ep_p.training())

        # (mu,lambda)-EP
        ep_ = EP_(DNA_SIZE=2,
                  DNA_BOUND=[-5.5, 5.5],
                  GENERATIONS=60,
                  POP_SIZE=100,
                  OFF_SIZE=200)
        ep__pops.append(ep_.training())

    np.savez("ES.npz",
             es_p_pops=es_p_pops,
             es__pops=es__pops,
             ep_p_pops=ep_p_pops,
             ep__pops=ep__pops)
    es_p_mean = mean(es_p_pops)
    es_p_max = max_(es_p_pops)
    es__mean = mean(es__pops)
    es__max = max_(es__pops)
    ep_p_mean = mean(ep_p_pops)
    ep_p_max = max_(ep_p_pops)
    ep__mean = mean(ep__pops)
    ep__max = max_(ep__pops)

    x = np.arange(0, 60)
    l1, = plt.plot(x, es_p_mean, color='red')
    l2, = plt.plot(x, es__mean, color='orange')
    l3, = plt.plot(x, ep_p_mean, color='blue')
    l4, = plt.plot(x, ep__mean, color='green')

    l5, = plt.plot(x, es_p_max, color='red', linestyle='--')
    l6, = plt.plot(x, es__max, color='orange', linestyle='--')
    l7, = plt.plot(x, ep_p_max, color='blue', linestyle='--')
    l8, = plt.plot(x, ep__max, color='green', linestyle='--')

    plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7, l8],
               labels=[u'mean:(μ+λ)-ES',
                       u'mean:(μ,λ)-ES',
                       u'mean:(μ+λ)-EP',
                       u'mean:(μ,λ)-EP',
                       u'max:(μ+λ)-ES',
                       u'max:(μ,λ)-ES',
                       u'max:(μ+λ)-EP',
                       u'max:(μ,λ)-EP'],
               loc='best')

    # plt.legend(handles=[l1, l3, l5, l7],
    #            labels=[u'mean:(μ+λ)-ES',
    #                    u'mean:(μ+λ)-EP',
    #                    u'max:(μ+λ)-ES',
    #                    u'max:(μ+λ)-EP'],
    #            loc='best')

    plt.xlabel('Generation')
    plt.ylabel('Normalized Fitness')
    plt.show()


def two():
    ep_p = EP_P(DNA_SIZE=2,
              DNA_BOUND=[-5.5, 5.5],
              GENERATIONS=100,
              POP_SIZE=50,
              OFF_SIZE=100)
    data = ep_p.training()
    drawing(data)


two()
