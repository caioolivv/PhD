#include "numcosmo.h"

int
main ()
{
  ncm_cfg_init ();

  NcGalaxySDPositionSRDY1 *rz_dist = nc_galaxy_sd_position_srd_y1_new ();
  NcGalaxySDZProxyGauss *zp_dist = nc_galaxy_sd_z_proxy_gauss_new ();
  NcGalaxySDShapeGauss *s_dist = nc_galaxy_sd_shape_gauss_new ();

  NcmVector *z_lim = ncm_vector_new (2);
  ncm_vector_set (z_lim, 0, 0.3);
  ncm_vector_set (z_lim, 1, 4.1);
  NcmVector *r_lim = ncm_vector_new (2);
  ncm_vector_set (r_lim, 0, 0.6);
  ncm_vector_set (r_lim, 1, 3.2);
  gdouble zp_sigma = 0.05;
  gdouble s_sigma = 0.05;

  nc_galaxy_sd_position_srd_y1_set_z_lim (rz_dist, z_lim);
  nc_galaxy_sd_position_srd_y1_set_r_lim (rz_dist, r_lim);
  nc_galaxy_sd_z_proxy_gauss_set_z_lim (zp_dist, z_lim);
  nc_galaxy_sd_z_proxy_gauss_set_sigma (zp_dist, zp_sigma);
  nc_galaxy_sd_shape_gauss_set_sigma (s_dist, s_sigma);

  NcGalaxyWLLikelihood *gwll = nc_galaxy_wl_likelihood_new (NC_GALAXY_SD_SHAPE (s_dist), NC_GALAXY_SD_Z_PROXY (zp_dist), NC_GALAXY_SD_POSITION (rz_dist));

  NcHICosmo *cosmo = nc_hicosmo_new_from_name (NC_TYPE_HICOSMO, "NcHICosmoLCDM");
  NcHaloDensityProfile *dp = NC_HALO_DENSITY_PROFILE (nc_halo_density_profile_nfw_new (NC_HALO_DENSITY_PROFILE_MASS_DEF_CRITICAL, 200.0));
  NcDistance *dist = nc_distance_new (4.0);
  nc_distance_prepare (dist, cosmo);
  NcWLSurfaceMassDensity *smd = nc_wl_surface_mass_density_new (dist);

  nc_galaxy_wl_likelihood_prepare (gwll, cosmo, dp, smd, 0.4);

  NcmStatsDistKDE *kde = nc_galaxy_wl_likelihood_peek_kde (gwll);
  NcmVector *x = ncm_vector_new (3);

  ncm_vector_set (x, 0, 1.6);
  ncm_vector_set (x, 1, 2.5);
  ncm_vector_set (x, 2, 0.001);

  gdouble p = ncm_stats_dist_eval (NCM_STATS_DIST (kde), x);

  printf("%f\n", p);
}
