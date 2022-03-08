"""
Uses the genetic algorithm to solve PSET Question 2
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt


class Process:
    def __init__(self, name, time):
        self.name = name
        self.time = time
        self.kernel = -1 # Either 0 or 1. -1 means uninitialized.


"""
An arrangement is one order of how different items should be put.
Is a simple 1-d array. Index corresponds to the index item in GeneticProcessAllocator.p_array.
Value corresponds to which kernel each process is supposed to be placed in
"""
class Arrangement:
    def __init__(self, array):
        self.array = array
        self.time = None
        self.k_dist = None

    def __repr__(self) -> str:
        string = "".join(self.array.astype(str))
        return f"<Arrangement {string}, time: {self.time}>"
    
    def __len__(self):
        return len(self.array)


class Colony:
    def __init__(self, genomes):
        self.genomes = genomes
        self.epoch = 0

    def get_best_time(self):
        return self.genomes[0].time

    def __getitem__(self, item):
        return self.genomes[item]


class GeneticProcessAllocator:

    def __init__(self, path=None, n_processes=100, linear=True):
        # Define allocator constants
        self.p_array = []
        self.kernel_num = 2
        self.processes = n_processes
        self.colony_size = 50
        self.epochs = 100

        # Runtime varying conditions
        self.elitism_ratio = 0.2
        self.mutation_ratio = 0.5
        self.mut_gene_prop = 0.5
        self.reproduce_ratio = 0.5

        self.colony = None

        if path:
            self.init_from_csv(path)
        else:
            self.init_from_generator(n_processes, linear)

    """
    Get process data from an existing CSV
    """
    def init_from_csv(self, path):
        # read_csv
        with open(path) as file:
            lines = file.readlines()
            for l in lines[1:]:
                name, time = l.split(',')
                self.p_array.append(Process(name, float(time)))
        self.processes = len(self.p_array)

    """
    Create a new set of process data using a generating distribution
    """
    def init_from_generator(self, n_processes, linear=True):
        # Gen according to n_processes
        # create process data
        if linear:
            processes = np.array(list(range(1, n_processes + 1)))
        else:
            randoms = (np.random.rand(n_processes)) * 10
            processes = np.round(np.power(2, randoms), 3)
        
        # write to .csv + Process Array
        with open("process_data.csv", 'w') as file:
            # write header
            file.write(f"p_id,p_time\n")
            for line in range(n_processes):
                p_name = f"process{line}"
                file.write(f"{p_name},{processes[line]}\n")
                self.p_array.append(Process(p_name, processes[line]))

    """
    Run self.epochs epochs of genetic evolution and prints results onto terminal
    """
    def run(self, print_every=10, graph=False, break_threshold=None):
        graph_results = []
        self.colony = self.create_colony()
        repeated = 0
        for gen in range(self.epochs):
            last_best = self.colony.get_best_time()
            self.colony = self.evolve()
            now_best = self.colony.get_best_time()
            if last_best == now_best:
                repeated += 1
                if break_threshold and repeated == break_threshold:
                    break
            else:
                repeated = 0
            if print_every and (gen % print_every == 0):
                print("="*30)
                print(f"Epoch {gen + 1}\n")
                for i in range(3):
                    print(i + 1, self.colony.genomes[i].time)
            if graph:
                graph_results.append(deepcopy(self.colony.get_best_time()))
        if graph:
            plt.title("Results over epochs")
            plt.plot(list(range(len(graph_results))), graph_results)
            plt.xlabel("Epoch")
            plt.ylabel("Top allocation time")
            plt.show()
        return self.colony

    def create_colony(self, size=None):
        size = size if size else self.colony_size
        self.colony = [Arrangement(np.random.randint(0, self.kernel_num, size=self.processes)) for x in range(size)]
        return Colony(self.colony)

    """
    Evaluates the 'fitness' for a single Arrangement. Is the max-time for any kernel.
    Modifies Arrangement.time and returns original Arrangement
    """
    def kernel_fitness_function(self, arrangement: Arrangement):
        kernels = dict()
        for index, p in enumerate(self.p_array):
            k_num = arrangement.array[index]
            if k_num != -1:
                if k_num not in kernels:
                    kernels[k_num] = p.time
                else:
                    kernels[k_num] += p.time
        k_max = max(kernels, key=lambda x: kernels[x])
        arrangement.k_dist = kernels
        arrangement.time = round(kernels[k_max], 5)
        return arrangement

    """
    Evolve 'colony' to the next stage, through using (1) elitism, (2) mutation, (3) reproduction
    By default will evolve self.colony
    """
    def evolve(self):
        colony = self.colony.genomes
        # First, get fitness function
        ranking = sorted([self.kernel_fitness_function(arr) for arr in colony], key=lambda x: x.time)
        # determine how many elitism, how many mutation, how many reproduced
        elitism_n = int(self.elitism_ratio * len(colony))
        mutation_n = int((len(colony) - elitism_n) / 2)
        reproduce_n = len(colony) - elitism_n - mutation_n
        # elitism
        new_colony = ranking[:elitism_n]
        # mutation_n
        new_colony += [self.mutate(deepcopy(random.choice(ranking[:int(self.mutation_ratio * len(colony))]))) 
                        for x in range(mutation_n)]
        # reproduce
        for n in range(reproduce_n):
            parent1 = random.choice(ranking[:int(self.reproduce_ratio * len(colony))])
            parent2 = random.choice(ranking[:int(self.reproduce_ratio * len(colony))])
            split = random.randint(0, len(parent1))
            child_arr = np.concatenate((parent1.array[:split], parent2.array[split:]))
            new_colony.append(Arrangement(child_arr))

        self.colony.genomes = sorted([self.kernel_fitness_function(arr) for arr in new_colony], key=lambda x: x.time)
        self.colony.epoch += 1
        return self.colony

    def mutate(self, arr):
        data = arr.array
        mut_p = int(self.mut_gene_prop * len(data))
        index_choices = list(range(len(data)))
        for i in range(mut_p):
            choice = random.choice(index_choices)
            index_choices.remove(choice)
            data[choice] = (data[choice] + random.randint(1, self.kernel_num - 1)) % self.kernel_num
        return arr


if __name__ == "__main__":
    gpe = GeneticProcessAllocator(None, 1000)
    gpe.run(graph=True)
