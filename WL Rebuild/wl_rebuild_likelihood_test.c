#include "numcosmo.h"

int
main ()
{
  ncm_cfg_init ();

  NcGalaxySDPositionFlat *pos_dist = nc_galaxy_sd_position_flat_new ();
  NcGalaxySDZProxyGauss *zp_dist = nc_galaxy_sd_z_proxy_gauss_new ();
  NcGalaxySDShapeGauss *s_dist = nc_galaxy_sd_shape_gauss_new ();

  NcmVector *z_lim = ncm_vector_new (2);
  ncm_vector_set (z_lim, 0, 0.6);
  ncm_vector_set (z_lim, 1, 2.9);
  NcmVector *r_lim = ncm_vector_new (2);
  ncm_vector_set (r_lim, 0, 0.3);
  ncm_vector_set (r_lim, 0, 2.0);
  gdouble zp_sigma = 1.0;
  gdouble s_sigma = 0.05;

  nc_galaxy_sd_position_flat_set_z_lim (pos_dist, z_lim);
  nc_galaxy_sd_position_flat_set_r_lim (pos_dist, r_lim);
  nc_galaxy_sd_z_proxy_gauss_set_z_lim (zp_dist, z_lim);
  nc_galaxy_sd_z_proxy_gauss_set_sigma (zp_dist, zp_sigma);
  nc_galaxy_sd_shape_gauss_set_sigma (s_dist, s_sigma);

  NcGalaxyWLLikelihood *gwll = nc_galaxy_wl_likelihood_new (NC_GALAXY_SD_SHAPE (s_dist), NC_GALAXY_SD_Z_PROXY (zp_dist), NC_GALAXY_SD_POSITION (pos_dist));

  NcmMatrix *obs = ncm_matrix_new (2, 3);

  ncm_matrix_set (obs, 0, 0, 1.5);
  ncm_matrix_set (obs, 0, 1, 0.8);
  ncm_matrix_set (obs, 0, 2, 0.3);
  ncm_matrix_set (obs, 1, 0, 1.9);
  ncm_matrix_set (obs, 1, 1, 1.8);
  ncm_matrix_set (obs, 1, 2, 0.15);

  nc_galaxy_wl_likelihood_set_obs (gwll, obs);

  NcHICosmo *cosmo = nc_hicosmo_new_from_name (NC_TYPE_HICOSMO, "NcHICosmoLCDM");
  NcHaloDensityProfile *dp = NC_HALO_DENSITY_PROFILE (nc_halo_density_profile_nfw_new (NC_HALO_DENSITY_PROFILE_MASS_DEF_CRITICAL, 200.0));
  NcDistance *dist = nc_distance_new (3.0);
  nc_distance_prepare (dist, cosmo);
  NcWLSurfaceMassDensity *smd = nc_wl_surface_mass_density_new (dist);

  nc_galaxy_wl_likelihood_prepare (gwll, cosmo, dp, smd, 0.4);
  gdouble res = nc_galaxy_wl_likelihood_kde_eval_m2lnP (gwll, cosmo, dp, smd, 0.4);

  printf ("%f\n", res);
}
