import subprocess

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy import stats
from Tools.Plotter import Plotter
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    subprocess.call(['python', 'Simulations\\basic_simulation.py'])