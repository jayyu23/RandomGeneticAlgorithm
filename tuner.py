from algorithm import GeneticProcessAllocator
import numpy as np
from matplotlib import pyplot as plt
import time
import multiprocessing as mp
import os

gpa = GeneticProcessAllocator(None, 100, True)
gpa.epochs = 200
gpa.colony_size = 50
SAVE_DIR = str()

plot_data = dict()  # name -> (metadata, x, y)

def vary_attribute(attr_name, low_bound=0, upper_bound=1, num=20, avg=1, remove_anom=0):
    vary_x = np.round(np.linspace(low_bound, upper_bound, num), 3)
    results = np.zeros(num)
    for times in range(avg):
        # print(times)
        curr = []
        for x in vary_x:
            print(x)
            setattr(gpa, attr_name, x)
            curr.append(gpa.run(print_every=None, graph=True, break_threshold=30)[0].time)
        results += np.array(curr)
        print(results)
    results /= avg
    results = np.round(results, 4)
    # Remove anomalies
    for n in range(remove_anom):
        m = np.max(results)
        index = np.where(results == m)[0][0]
        results = np.delete(results, index)
        vary_x = np.delete(vary_x, index)

    r_coef = np.round(np.corrcoef(vary_x, results)[1][0], 3)
    poly_fit = np.polyfit(vary_x, results, 2)
    plt.title(f"Varying {attr_name} (r={r_coef})")
    plt.xlabel(f"Value of {attr_name}")
    plt.ylabel(f"Top allocation time ({gpa.epochs} epochs)")
    plt.scatter(vary_x, results)
    plt.plot(vary_x, np.poly1d(poly_fit)(vary_x), color='red')
    plt.savefig(os.path.join(SAVE_DIR, f"{attr_name}.png"))
    plt.show()
    plt.close()


def vary_time(attr_name, low_bound=0.05, upper_bound=0.95, num=100, avg=20, converge_epoch=30):
    vary_x = np.round(np.linspace(low_bound, upper_bound, num), 3)
    results = np.zeros(num)
    for times in range(avg):
        print(times)
        cur = []
        for x in vary_x:
            setattr(gpa, attr_name, x)
            run_result = gpa.run(print_every=None, graph=False, break_threshold=converge_epoch)
            # print(run_result.epoch)
            cur.append(run_result.epoch - converge_epoch)
        results += np.array(cur)
        # print(cur)
    results /= avg
    results = np.round(results, 4)
    # results = np.array(results)
    meta_data = (num, avg, converge_epoch)
    plot_data[attr_name] = (meta_data, vary_x, results)


def plot_results():
    print(plot_data)
    for name, data in plot_data.items():
        meta_data, x, y = data
        num, avg, converge_epoch = meta_data
        r_coef = np.round(np.corrcoef(x, y)[1][0], 3)
        poly_fit = np.polyfit(x, y, 2)
        plt.title(f"Varying {name} (avg of {avg} runs)")
        plt.xlabel(f"Value of {name} (n={num})")
        plt.ylabel(f"Convergence time (C={converge_epoch})")
        plt.scatter(x, y)
        plt.plot(x, np.poly1d(poly_fit)(x), color='red')
        plt.savefig(os.path.join(SAVE_DIR, f"{name}.png"))
        plt.show()
        plt.close()


if __name__ == "__main__":
    start = time.time()
    SAVE_DIR = os.path.join("runs", str(time.time_ns()))
    os.mkdir(SAVE_DIR)
    attrs = ["elitism_ratio", "mutation_ratio", "mut_gene_prop", "reproduce_ratio"]
    processes = []
    for v in attrs:
        p = mp.Process(target=vary_time, args=(v,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    plot_results()
    stop = time.time()
    print(f"Time taken: {round(stop - start, 4)}")