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

# np.random.seed(0)
# Define cosmological parameters
cosmo = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)

cluster_m     = 1.e15 # Cluster mass
cluster_z     = 0.4   # Cluster redshift
concentration = 4     # Concentrion parameter NFW profile
ngals         = 10000 # Number of galaxies
Delta         = 200   # Overdensity parameter definition NFW profile
cluster_ra    = 0.0   # Cluster right ascension
cluster_dec   = 0.0   # Cluster declination
shapenoise    = 5e-2  # True ellipticity standard variation
photoz_noise  = 5e-2
ndata         = 10000

# Create galaxy catalog and Cluster object

def create_nc_data_cluster_wll (theta, g_t, z_source, z_cluster, cosmo, dist, sigma_z = None, sigma_g = None):
    r = clmm.convert_units (theta, "radians", "Mpc", redshift = z_cluster, cosmo = cosmo)
    obs = Ncm.Matrix.new (len (theta), 3)

    for i in range (len (theta)):
        obs.set (i, 0, r[i])
        obs.set (i, 1, z_source[i])
        obs.set (i, 2, g_t[i])

    # gsdp  = Nc.GalaxySDPositionFlat ()
    gsdp  = Nc.GalaxySDPositionSRDY1 ()
    gsdzp = Nc.GalaxySDZProxyGauss ()
    gsds  = Nc.GalaxySDShapeGauss ()

    gsdp.set_z_lim (Ncm.Vector.new_array ([0.3, 4.1]))
    gsdp.set_r_lim (Ncm.Vector.new_array ([0.6, 3.2]))
    gsdzp.set_z_lim (Ncm.Vector.new_array ([0.4, 4.2]))
    gsdzp.set_sigma (sigma_z)
    gsds.set_sigma (sigma_g)

    gwll = Nc.GalaxyWLLikelihood (s_dist=gsds, zp_dist=gsdzp, rz_dist=gsdp)
    gwll.set_obs (obs)
    gwll.set_ndata (ndata)
    gwll.set_cut (z_cluster + 0.1, 4.0, 0.75, 3.0)

    ga = Ncm.ObjArray.new ()
    ga.add (gwll)

    dcwll = Nc.DataClusterWLL (galaxy_array=ga, z_cluster=z_cluster)
    dcwll.set_init (True)
    
    return dcwll, gwll

def create_fit_obj (data_array, mset):
    dset = Ncm.Dataset.new ()
    for data in data_array:
        dset.append_data (data)
    lh = Ncm.Likelihood.new (dset)
    fit = Ncm.Fit.new (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)

    return fit

# np.random.seed(10)

moo = clmm.Modeling (massdef='mean', delta_mdef=200, halo_profile_model='nfw')
moo.set_cosmo(cosmo)
mset = moo.get_mset ()

MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
mset.prepare_fparam_map ()

data = mock.generate_galaxy_catalog (cluster_m, cluster_z, concentration, cosmo, "chang13", zsrc_min = cluster_z + 0.1, shapenoise=shapenoise, ngals=ngals, cluster_ra=cluster_ra, cluster_dec=cluster_dec, photoz_sigma_unscaled=photoz_noise)
gc = clmm.GalaxyCluster("CL_noisy_z", cluster_ra, cluster_dec, cluster_z, data)
gc.compute_tangential_and_cross_components(geometry="flat")

ggt, gwll = create_nc_data_cluster_wll (gc.galcat['theta'], gc.galcat['et'], gc.galcat['z'], cluster_z, cosmo, cosmo.dist, sigma_z=photoz_noise, sigma_g=shapenoise)
fit = create_fit_obj ([ggt], mset)

kde = gwll.peek_kde ()
# dist = NCM_STATS_DIST (kde)

r = np.linspace (0.6, 3.2, 1000)
p = []

for i in range (1):
    data = Ncm.Vector.new (3)
    data.set (0, r[i])
    data.set (1, 1.5)
    data.set (2, 0.02)

    p.append (kde.eval (data))
