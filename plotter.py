#!/usr/bin/env python3
#
# modify as you see fit.
#
# This script was written for red hat based systems, so
# depending on your system you may need to install
# matplotlip and pandas "pip install matplotlib --user"
# You may also want to use virtual environments
#
#
# Usage
#
# ./plotter.py plotname.png perf1.csv [perf2.csv perf3.csv ....]
#
# This will create a png of all of the performance results.


import sys

import pandas as pd
from matplotlib import pyplot as plt

file_names  = sys.argv[2:]
output_file = sys.argv[1]

# See: https://aiichironakano.github.io/cs596/PeakFlops.pdf
# https://www.cpu-world.com/CPUs/Xeon/Intel-Xeon%20E3-1240%20v5.html
#
# microarchitecture diagrams for skylake
# https://www.cpu-world.com/CPUs/Xeon/Intel-Xeon%20E3-1240%20v5.html
# https://chipsandcheese.com/p/skylake-intels-longest-serving-architecture

# All of these should really be getopts
number_of_cores=1 # we can look at the one core performance if we are not using multiple threads
ghz_freq_of_core=2.2
num_fma_per_core=2  # You need to look at the microarchitecture and pull out a block diagram
width_fma=8         # AVX2 for floats is 8-floats wide

flop_per_cycle=2*width_fma # fma is 2 flops for multiply and add
machine_gflop_peak = number_of_cores*ghz_freq_of_core*flop_per_cycle



for fn in file_names:
    df = pd.read_csv(fn)
    plt.plot(df['size'], df['throughput'])
plt.legend(file_names, loc="upper right")
plt.ylim(0, machine_gflop_peak)
plt.xlim(0,None)

# Feel free to edit these or make them arguments
plt.title("Performance Versus Problem Size")
plt.xlabel("Problem Size")
plt.ylabel("Performance (GFLOP/s)")

#plt.show()
plt.savefig(output_file)

