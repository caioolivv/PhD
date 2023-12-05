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

# plt.rcParams['text.latex.preamble'] = [r'\usepackage{pxfonts, mathpazo}']
plt.rc('text', usetex=True)

np.random.seed(0)

# Define cosmological parameters
cosmo_clmm = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)

cluster_m     = 10**(14.301) # Cluster mass
cluster_z     = 0.4          # Cluster redshift
concentration = 4            # Concentrion parameter NFW profile
ngals         = 10000        # Number of galaxies
Delta         = 200          # Overdensity parameter definition NFW profile
cluster_ra    = 0.0          # Cluster right ascension
cluster_dec   = 0.0          # Cluster declination
sigma_g       = 5e-2         # True ellipticity standard variation
sigma_z       = 5e-2         # True redshift standard variation
ndata         = 1000         # Number of data points on KDE
prec          = 14           # -log10 of the integral precision

is_kde        = False        # Use KDE or not

moo = clmm.Modeling (massdef='mean', delta_mdef=200, halo_profile_model='nfw')
moo.set_cosmo(cosmo_clmm)
mset = moo.get_mset ()

MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
mset.prepare_fparam_map ()

# Create data
data = mock.generate_galaxy_catalog (cluster_m, cluster_z, concentration, cosmo_clmm, "chang13", zsrc_min = cluster_z + 0.1, shapenoise=sigma_g, ngals=ngals, cluster_ra=cluster_ra, cluster_dec=cluster_dec, photoz_sigma_unscaled=sigma_z)
gc = clmm.GalaxyCluster("CL_noisy_z", cluster_ra, cluster_dec, cluster_z, data)
gc.compute_tangential_and_cross_components(geometry="flat")
r = clmm.convert_units (gc.galcat['theta'], "radians", "Mpc", redshift = cluster_z, cosmo = cosmo_clmm)

obs = Ncm.Matrix.new (len (gc.galcat['theta']), 3)

for i in range (len (gc.galcat['theta'])):
    obs.set (i, 0, r[i])
    obs.set (i, 1, gc.galcat['z'][i])
    obs.set (i, 2, gc.galcat['et'][i])
    # obs.set (i, 2, gc.galcat['ex'][i])

cosmo = Nc.HICosmo.new_from_name(Nc.HICosmo, "NcHICosmoLCDM")
dp    = Nc.HaloDensityProfileNFW.new(Nc.HaloDensityProfileMassDef.CRITICAL, 200)
dist  = Nc.Distance.new(10)

dist.prepare(cosmo)

smd = Nc.WLSurfaceMassDensity.new(dist)
rng = Ncm.RNG.seeded_new("mt19937", 1)

gsdp  = Nc.GalaxySDPositionSRDY1()
gsdzp = Nc.GalaxySDZProxyGauss()
gsds  = Nc.GalaxySDShapeGauss()

gsdp.set_r_lim(Ncm.Vector.new_array([1e-6, 4]))
gsdp.set_z_lim(Ncm.Vector.new_array([1e-6, 10]))
gsdzp.set_z_lim(Ncm.Vector.new_array([1e-6, 10]))
gsdzp.set_sigma(sigma_z)
gsds.set_sigma(sigma_g)

gwll = Nc.GalaxyWLLikelihood(s_dist=gsds, zp_dist=gsdzp, rz_dist=gsdp)

obs_matrix = Ncm.Matrix.new(ngals, 3)

for i in range(ngals):
    vec = gwll.gen(cosmo, dp, smd, cluster_z, rng)
    for j in range(3):
        obs_matrix.set(i, j, vec.get(j))
    #     print(j, obs.get(j))
    # print('')

gwll.set_obs (obs_matrix)
gwll.set_ndata (ndata)
gwll.set_cut (3.0, 5.0)
gwll.set_prec (10**(-prec))

ga = Ncm.ObjArray.new ()
ga.add (gwll)

dcwll = Nc.DataClusterWLL (galaxy_array=ga, z_cluster=cluster_z)
dcwll.set_init (True)
dcwll.set_kde (is_kde)

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

fit.run (Ncm.FitRunMsgs.SIMPLE)
fit.obs_fisher ()
fit.log_info ()
fit.log_covar ()

Ncm.func_eval_set_max_threads (12)
Ncm.func_eval_log_pool_stats ()

init_sampler = Ncm.MSetTransKernGauss.new (0)
init_sampler.set_mset (mset)
init_sampler.set_prior_from_mset ()
init_sampler.set_cov_from_rescale (1.0e-1)

nwalkers = 200
stretch = Ncm.FitESMCMCWalkerAPES.new (nwalkers, mset.fparams_len ())
esmcmc  = Ncm.FitESMCMC.new (fit, nwalkers, init_sampler, stretch, Ncm.FitRunMsgs.SIMPLE)
esmcmc.set_auto_trim_div (100)
esmcmc.set_max_runs_time (2.0 * 60.0)

if is_kde:
    esmcmc.set_data_file (f"Fits/wl_rebuild_kde_{ndata}_{nwalkers}.fits")
else:
    esmcmc.set_data_file (f"Fits/wl_rebuild_integral_{prec}_{nwalkers}.fits")

esmcmc.set_nthreads(12)
esmcmc.start_run ()
esmcmc.run (100000/nwalkers)
esmcmc.end_run ()

mcat = esmcmc.peek_catalog ()


rows = np.array([mcat.peek_row(i).dup_array() for i in range(nwalkers * 400, mcat.len())])
params = ["$" + mcat.col_symb(i) + "$" for i in range (mcat.ncols())]

partial = ChainConsumer ()
partial.add_chain(rows[:,1:], parameters=params[1:], name=f"$\sigma_{{\epsilon^s}} = {sigma_g}$")
partial.configure(spacing=0.0, usetex=True, colors='#D62728', shade=True, shade_alpha=0.2, bar_shade=True, smooth=True, kde=True, legend_color_text=False, linewidths=2)

CC_fig = partial.plotter.plot(figsize=(8, 8), truth=[4, 14.301])

fig = plt.figure(num=CC_fig, figsize=(8,8), dpi=300, facecolor="white")

if is_kde:
    plt.title(f"KDE, {ndata} data points, {nwalkers} walkers")
    plt.savefig(f"Plots/wl_rebuild_kde_{ndata}_{nwalkers}.png", bbox_inches='tight', dpi=300)
else:
    plt.title(f"Integral, {prec} precision, {nwalkers} walkers")
    plt.savefig(f"Plots/wl_rebuild_integral_{prec}_{nwalkers}.png", bbox_inches='tight', dpi=300)