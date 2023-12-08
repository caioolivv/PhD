#!/usr/bin/python
import os
import sys
import gi

from numcosmo_py import Nc
from numcosmo_py import Ncm

__name__ = "NcContext"

Ncm.cfg_init ()
Ncm.cfg_set_log_handler (lambda msg: sys.stdout.write (msg) and sys.stdout.flush ())

import numpy as np
from numpy import random
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt

# plt.rcParams['text.latex.preamble'] = [r'\usepackage{pxfonts, mathpazo}']
plt.rc('text', usetex=True)

np.random.seed(0)
rng = Ncm.RNG.seeded_new("mt19937", 1)

# Define cosmological parameters
# cosmo_clmm = Cosmology(H0 = 70.0, Omega_dm0 = 0.27 - 0.045, Omega_b0 = 0.045, Omega_k0 = 0.0)

cluster_m     = 10**(14.301) # Cluster mass
cluster_z     = 0.4          # Cluster redshift
concentration = 4            # Concentrion parameter NFW profile
ngals         = 10000        # Number of galaxies
Delta         = 200          # Overdensity parameter definition NFW profile
sigma_g       = 5e-2         # True ellipticity standard variation
sigma_z       = 5e-2         # True redshift standard variation
ndata         = 1000         # Number of data points on KDE
prec          = 14           # -log10 of the integral precision

is_kde        = False        # Use KDE or not

cosmo = Nc.HICosmoLCDM.new()
dp    = Nc.HaloDensityProfileNFW.new(Nc.HaloDensityProfileMassDef.CRITICAL, 200)
dist  = Nc.Distance.new(10)

dist.prepare(cosmo)

smd = Nc.WLSurfaceMassDensity.new(dist)

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

mset = Ncm.MSet.empty_new()
mset.set(cosmo)
mset.set(dp)
mset.set(smd)

MDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:log10MDelta")
cDelta_pi = mset.param_get_by_full_name ("NcHaloDensityProfile:cDelta")

mset.param_set_ftype (MDelta_pi.mid, MDelta_pi.pid, Ncm.ParamType.FREE)
mset.param_set_ftype (cDelta_pi.mid, cDelta_pi.pid, Ncm.ParamType.FREE)
mset.prepare_fparam_map ()

dset = Ncm.Dataset.new ()
dset.append_data (dcwll)
lh = Ncm.Likelihood.new (dset)
fit = Ncm.Fit.factory (Ncm.FitType.NLOPT, "ln-neldermead", lh, mset, Ncm.FitGradType.NUMDIFF_FORWARD)

#fit.run (Ncm.FitRunMsgs.FULL)
#fit.obs_fisher ()
#fit.log_info ()
#fit.log_covar ()

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
    fig.savefig(f"Plots/wl_rebuild_kde_{ndata}_{nwalkers}.png", bbox_inches='tight')
else:
    fig.savefig(f"Plots/wl_rebuild_integral_{prec}_{nwalkers}.png", bbox_inches='tight')