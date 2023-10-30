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
from clmm.support import mock_data as mock
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

plt.rcParams['text.latex.preamble'] = [r'\usepackage{pxfonts, mathpazo}']
plt.rc('text', usetex=True)

np.random.seed(10)
# Define cosmological parameters
cosmo = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)

cluster_m     = 1.e15 # Cluster mass
cluster_z     = 0.4   # Cluster redshift
concentration = 4     # Concentrion parameter NFW profile
ngals         = 10000 # Number of galaxies
Delta         = 200   # Overdensity parameter definition NFW profile
cluster_ra    = 0.0   # Cluster right ascension
cluster_dec   = 0.0   # Cluster declination
sigma_g       = 5e-2  # True ellipticity standard variation
sigma_z       = 1e-2
ndata         = 1000
scale         = 1e-6


# Create Model Set
moo = clmm.Modeling (massdef='mean', delta_mdef=200, halo_profile_model='nfw')
moo.set_cosmo(cosmo)
mset = moo.get_mset ()

MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
mset.prepare_fparam_map ()

# Create data
data = mock.generate_galaxy_catalog (cluster_m, cluster_z, concentration, cosmo, "chang13", zsrc_min = cluster_z + 0.1, shapenoise=sigma_g, ngals=ngals, cluster_ra=cluster_ra, cluster_dec=cluster_dec, photoz_sigma_unscaled=sigma_z)
gc = clmm.GalaxyCluster("CL_noisy_z", cluster_ra, cluster_dec, cluster_z, data)
gc.compute_tangential_and_cross_components(geometry="flat")
r = clmm.convert_units (gc.galcat['theta'], "radians", "Mpc", redshift = cluster_z, cosmo = cosmo)

obs = Ncm.Matrix.new (len (gc.galcat['theta']), 3)

for i in range (len (gc.galcat['theta'])):
    obs.set (i, 0, r[i])
    obs.set (i, 1, gc.galcat['z'][i])
    obs.set (i, 2, gc.galcat['et'][i])

gsdp  = Nc.GalaxySDPositionSRDY1 ()
gsdzp = Nc.GalaxySDZProxyGauss ()
gsds  = Nc.GalaxySDShapeGauss ()

gsdp.set_z_lim (Ncm.Vector.new_array ([0.4, 4.5]))
gsdp.set_r_lim (Ncm.Vector.new_array ([0.4, 4.5]))
gsdzp.set_z_lim (Ncm.Vector.new_array ([0.4, 4.5]))
gsdzp.set_sigma (sigma_z)
gsds.set_sigma (sigma_g)

gwll = Nc.GalaxyWLLikelihood (s_dist=gsds, zp_dist=gsdzp, rz_dist=gsdp)
gwll.set_obs (obs)
gwll.set_ndata (ndata)
gwll.set_cut (cluster_z + 0.1, 4.0, 0.75, 3.0)
gwll.set_scale_cut (scale)

ga = Ncm.ObjArray.new ()
ga.add (gwll)

dcwll = Nc.DataClusterWLL (galaxy_array=ga, z_cluster=cluster_z)
dcwll.set_init (True)

dset = Ncm.Dataset.new ()
dset.append_data (dcwll)
lh = Ncm.Likelihood.new (dset)
fit = Ncm.Fit.new (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)

fit.run (Ncm.FitRunMsgs.SIMPLE)
fit.obs_fisher ()
fit.log_info ()
fit.log_covar ()