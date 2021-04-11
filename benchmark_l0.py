"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import pickle
import tensorflow as tf
import numpy as np
import os
from utils import functions, pretty_print
from utils.symbolic_network import SymbolicNetL0
from inspect import signature
import benchmark
import argparse

tf.compat.v1.disable_eager_execution()

N_TRAIN = 256       # Size of training dataset
N_VAL = 100         # Size of validation dataset
DOMAIN = (-1, 1)    # Domain of dataset
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])   # Use this format if each input variable has a different domain
N_TEST = 100        # Size of test dataset
DOMAIN_TEST = (-2, 2)   # Domain of test dataset - should be larger than training domain to test extrapolation
NOISE_SD = 0        # Standard deviation of noise for training dataset
var_names = ["x", "y", "z"]

# Standard deviation of random distribution for weight initializations.
init_sd_first = 0.5
init_sd_last = 0.5
init_sd_middle = 0.5


generate_data = benchmark.generate_data


class Benchmark(benchmark.Benchmark):
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""
    def __init__(self, results_dir, n_layers=3, reg_weight=1e-2, learning_rate=1e-2,
                 n_epochs1=20001, n_epochs2=10001):
        """Set hyper-parameters"""
        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            *[functions.Exp()] * 2, #something wrong with the exponent
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]

        self.n_layers = n_layers              # Number of hidden layers
        self.reg_weight = reg_weight     # Regularization weight
        self.learning_rate = learning_rate
        self.summary_step = 1000    # Number of iterations at which to print to screen
        self.n_epochs1 = n_epochs1
        self.n_epochs2 = n_epochs2

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        self.results_dir = results_dir

        # Save hyperparameters to file
        result = {
            "learning_rate": self.learning_rate,
            "summary_step": self.summary_step,
            "n_epochs1": self.n_epochs1,
            "n_epochs2": self.n_epochs2,
            "activation_funcs_name": [func.name for func in self.activation_funcs],
            "n_layers": self.n_layers,
            "reg_weight": self.reg_weight,
        }
        with open(os.path.join(self.results_dir, 'params.pickle'), "wb+") as f:
            pickle.dump(result, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=1e-2, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=20001, help="Number of epochs to train the first stage")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    import json

    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    bench.benchmark(lambda x: x, func_name="x", trials=5)
    bench.benchmark(lambda x: x**2, func_name="x^2", trials=20)
    bench.benchmark(lambda x: x**3, func_name="x^3", trials=20)
    bench.benchmark(lambda x: np.sin(2*np.pi*x), func_name="sin(2pix)", trials=20)
    bench.benchmark(lambda x: np.exp(x), func_name="e^x", trials=20)
    bench.benchmark(lambda x, y: x*y, func_name="xy", trials=5)
    bench.benchmark(lambda x, y: np.sin(2 * np.pi * x) + np.sin(4*np.pi * y),
                    func_name="sin(2pix)+sin(2py)", trials=20)
    bench.benchmark(lambda x, y, z: 0.5*x*y + 0.5*z, func_name="0.5xy+0.5z", trials=5)
    bench.benchmark(lambda x, y, z: x**2 + y - 2*z, func_name="x^2+y-2z", trials=20)
    bench.benchmark(lambda x: np.exp(-x**2), func_name="e^-x^2", trials=20)
    bench.benchmark(lambda x: 1 / (1 + np.exp(-10*x)), func_name="sigmoid(10x)", trials=20)
    bench.benchmark(lambda x, y: x**2 + np.sin(2*np.pi*y), func_name="x^2+sin(2piy)", trials=20)
    #
    # # 3-layer functions
    # bench.benchmark(lambda x, y, z: (x + y * z) ** 3, func_name="(x+yz)^3", trials=20)
