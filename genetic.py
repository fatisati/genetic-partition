import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

import pandas as pd


class GeneticPartition:
    def __init__(self, people_arr, k, can_split):
        self.people_arr = people_arr
        self.population_size = 300
        self.crossover_population_size = 20
        self.k = k
        self.partition_sum = np.sum(people_arr) / self.k
        self.split_rate = 0.4
        self.min_split_val = 10
        self.mutation_rate = 0.1
        print(self.partition_sum, np.sum(people_arr))
        self.best_scores = []
        self.can_split = can_split

    @staticmethod
    def permute_arr_cols(arr):
        for col_idx in range(arr.shape[1]):
            random.shuffle(arr[:, col_idx])
        return arr

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.permute_arr_cols(deepcopy(self.people_arr)))
        return population

    def fitness(self, sample):
        res = sum(abs(np.sum(sample, axis=1) - self.partition_sum))
        # res = 0
        # for part in sample:
        #     res += abs(sum(part) - self.partition_sum)
        # print(res)
        return res

    def selection(self, population):
        scores = []
        for i in range(len(population)):
            scores.append(self.fitness(population[i]))

        sorted_idx = list(np.array(scores).argsort())
        selected_pop = []
        for idx in sorted_idx:
            selected_pop.append(population[idx])
        return selected_pop[:self.crossover_population_size]

    def crossover(self, mother, father):
        return np.concatenate((mother[:, :self.k // 2], father[:, self.k // 2:]), axis=1)

    @staticmethod
    def random_change(col):
        src = random.randint(0, len(col) - 1)
        dst = random.randint(0, len(col) - 1)
        tmp = col[src]
        col[src] = col[dst]
        col[dst] = tmp
        return col

    @staticmethod
    def rand_split(n):
        k = int(0.5 * n)
        return k, n - k

    def mutation(self, sample):
        for col_idx in range(sample.shape[1]):
            rand = random.random()
            if rand < self.mutation_rate:
                sample[:, col_idx] = self.random_change(sample[:, col_idx])

            rand = random.random()
            col = sample[:, col_idx]

            if rand < self.split_rate and max(col) > self.min_split_val and can_split[col_idx] == 1:
                col_size = len(col)
                src = np.argmax(col)

                k, v = self.rand_split(col[src])
                dst = random.randint(0, col_size - 1)
                sample[:, col_idx][src] = k
                sample[:, col_idx][dst] += v

        return sample

    def run_(self, generation_cnt):
        initial_pop = self.generate_initial_population()
        best = ''
        best_fitness = sys.maxsize

        for i in range(generation_cnt):
            print(i)
            selected_pop = self.selection(initial_pop)
            next_gen = []

            for mother in selected_pop:
                for father in selected_pop:
                    child = self.crossover(mother, father)
                    child = self.mutation(child)
                    score = self.fitness(child)
                    if score < best_fitness:
                        best_fitness = score
                        best = child
                    next_gen.append(child)
            print(best_fitness)
            if best_fitness == 0:
                break
            self.best_scores.append(best_fitness)
            initial_pop = next_gen

        return best, best_fitness


def read_data():
    df = pd.read_excel('vam.xlsx', index_col=0).fillna(0).drop(['sum'], axis=1)
    can_split = df['split']
    df = df.drop(['split'], axis=1)
    print(df.T.shape)
    return df.T.values, df.index, can_split
# import os
# def save_file(df, loss, name):
#     if name in os.listdir('./')

if __name__ == '__main__':
    people_arr, names, can_split = read_data()
    gen_alg = GeneticPartition(people_arr, 10, can_split)
    best_ans, best_score = gen_alg.run_(250)

    print(best_ans, best_score)
    sums = np.sum(best_ans, axis=1)
    print(sums.shape)
    res_df = pd.DataFrame(best_ans)
    res_df.columns = names
    res_df['sum'] = sums
    res_df['abs diff'] = abs(gen_alg.partition_sum - sums)
    res_df.to_excel('best_ans_{0}.xlsx'.format(best_score))

    plt.plot(gen_alg.best_scores)
    plt.savefig('error.png')
    # plt.show()
