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

import clmm
import numpy as np
from numpy import random
from clmm import Cosmology
# from clmm.support import mock_data as mock
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

# plt.rcParams['text.latex.preamble'] = [r'\usepackage{pxfonts, mathpazo}']
plt.rc('text', usetex=True)

np.random.seed(0)
# Define cosmological parameters
cosmo_clmm = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)

cluster_m     = 1.e15 # Cluster mass
cluster_z     = 0.4   # Cluster redshift
concentration = 4     # Concentrion parameter NFW profile
ngals         = 10000 # Number of galaxies
Delta         = 200   # Overdensity parameter definition NFW profile
cluster_ra    = 0.0   # Cluster right ascension
cluster_dec   = 0.0   # Cluster declination
sigma_g       = 5e-2  # True ellipticity standard variation
sigma_z       = 5e-2
ndata         = 10000

cosmo = Nc.HICosmo.new_from_name(Nc.HICosmo, "NcHICosmoLCDM")
dp    = Nc.HaloDensityProfileNFW.new(Nc.HaloDensityProfileMassDef.CRITICAL, 200)
dist  = Nc.Distance.new(6)

dist.prepare(cosmo)

smd = Nc.WLSurfaceMassDensity.new(dist)
rng = Ncm.RNG.seeded_new("mt19937", 1)

gsdp  = Nc.GalaxySDPositionSRDY1()
gsdzp = Nc.GalaxySDZProxyGauss()
gsds  = Nc.GalaxySDShapeGauss()

gsdp.set_z_lim(Ncm.Vector.new_array([1e-6, 4.5]))
gsdp.set_r_lim(Ncm.Vector.new_array([1e-6, 4.5]))
gsdzp.set_z_lim(Ncm.Vector.new_array([1e-6, 4.5]))
gsdzp.set_sigma(sigma_z)
gsds.set_sigma(sigma_g)

gwll = Nc.GalaxyWLLikelihood(s_dist=gsds, zp_dist=gsdzp, rz_dist=gsdp)

obs_matrix = Ncm.Matrix.new(ngals, 3)

for i in range(ngals):
    obs = gwll.gen(cosmo, dp, smd, cluster_z, rng)
    for j in range(3):
        obs_matrix.set(i, j, obs.get(j))
    #     print(j, obs.get(j))
    # print('')

gwll.set_obs (obs_matrix)
gwll.set_ndata (ndata)
gwll.set_cut (0.75, 3)

ga = Ncm.ObjArray.new ()
ga.add (gwll)

dcwll = Nc.DataClusterWLL (galaxy_array=ga, z_cluster=cluster_z)
dcwll.set_init (True)

moo = clmm.Modeling (massdef='critical', delta_mdef=200, halo_profile_model='nfw')
moo.set_cosmo(cosmo_clmm)
mset = moo.get_mset ()

MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
mset.prepare_fparam_map ()

dset = Ncm.Dataset.new ()
dset.append_data (dcwll)
lh = Ncm.Likelihood.new (dset)
fit = Ncm.Fit.new (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)

fit.run (Ncm.FitRunMsgs.FULL)
fit.obs_fisher (Ncm.FitRunMsgs.FULL)
fit.log_info ()
fit.log_covar ()
# fit.fisher()