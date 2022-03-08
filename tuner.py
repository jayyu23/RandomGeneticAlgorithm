from algorithm import GeneticProcessAllocator
import numpy as np
from matplotlib import pyplot as plt
import time
import os

gpa = GeneticProcessAllocator(None, 100, True)
gpa.epochs = 200
gpa.colony_size = 50
SAVE_DIR = str()


def vary_attribute(attr_name, low_bound=0, upper_bound=1, num=30, avg=1, remove_anom=0):
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


def vary_time(attr_name, low_bound=0, upper_bound=1, num=30, avg=3, converge_epoch=50):
    vary_x = np.round(np.linspace(low_bound, upper_bound, num), 3)
    results = np.zeros(num)
    for times in range(avg):
        cur = []
        for x in vary_x:
            setattr(gpa, attr_name, x)
            run_result = gpa.run(print_every=None, graph=False, break_threshold=converge_epoch)
            # print(run_result.epoch)
            cur.append(run_result.epoch)
        results += np.array(cur)
    results /= avg
    results = np.round(results, 4)
    # results = np.array(results)
    print(results)
    r_coef = np.round(np.corrcoef(vary_x, results)[1][0], 3)
    poly_fit = np.polyfit(vary_x, results, 2)
    plt.title(f"Varying {attr_name} (r={r_coef})")
    plt.xlabel(f"Value of {attr_name}")
    plt.ylabel(f"Convergence time (epochs)")
    plt.scatter(vary_x, results)
    plt.plot(vary_x, np.poly1d(poly_fit)(vary_x), color='red')
    plt.savefig(os.path.join(SAVE_DIR, f"{attr_name}.png"))
    plt.show()
    plt.close()


if __name__ == "__main__":
    SAVE_DIR = os.path.join("runs", str(time.time_ns()))
    os.mkdir(SAVE_DIR)
    # vary_time("elitism_ratio")
    # vary_time("mutation_ratio")
    vary_time("mut_gene_prop")
    # vary_time("reproduce_ratio")