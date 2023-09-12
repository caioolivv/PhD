# Importing NumCosmo
import os
import sys
import gi

gi.require_version('NumCosmo', '1.0')
gi.require_version('NumCosmoMath', '1.0')
from gi.repository import GObject
from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

os.environ['CLMM_MODELING_BACKEND'] = 'nc'

__name__ = "NcContext"

Ncm.cfg_init ()
Ncm.cfg_set_log_handler (lambda msg: sys.stdout.write (msg) and sys.stdout.flush ())

#Importing Numpy and SciPy
import numpy as np
import scipy as sp
from numpy import random

# Setting seed
np.random.seed(0)

# Defining PDFs
def gaussian_pdf(x):
    return np.exp(-(x-6)**2 / 2) / (np.sqrt(2*np.pi))

def gamma_pdf(x):
    return (x * np.exp(-x)) / sp.special.gamma(2)

def gaussian_gamma_pdf(x):
    return np.add(gaussian_pdf(x), gamma_pdf(x))

# Creating random samples
ndata = 1000

g_samples = []
y_samples = []
gy_samples = []

for i in range(ndata):
    g_samples.append(np.random.normal(6))
    y_samples.append(np.random.gamma(2))
    gy_samples.append(np.random.normal(6) + np.random.gamma(2))

# Creating control KDE with RoT
kernel_control = Ncm.StatsDistKernelGauss.new(3)
kde_control    = Ncm.StatsDistKDE.new(kernel_control, Ncm.StatsDistCV.NONE)

for i in range(ndata):
    kde_control.add_obs(Ncm.Vector.new_array([g_samples[i], y_samples[i], gy_samples[i]]))

kde_control.prepare()

# Creating test KDE with LOO
kernel_test = Ncm.StatsDistKernelGauss.new(3)
kde_test    = Ncm.StatsDistKDE.new(kernel_test, Ncm.StatsDistCV.LOO)

for i in range(ndata):
    kde_test.add_obs(Ncm.Vector.new_array([g_samples[i], y_samples[i], gy_samples[i]]))

kde_test.prepare()