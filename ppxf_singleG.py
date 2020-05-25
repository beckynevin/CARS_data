from __future__ import print_function

import sys
import os.path
from astropy.io import fits

from time import clock
from scipy import ndimage
from scipy import interpolate
import numpy as np
import glob

from matplotlib import pyplot as plt 
from matplotlib.pyplot import cm 
from astropy.convolution import convolve, Box1DKernel

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from ppxf import ppxf
import ppxf.ppxf_util as util

# Print everything
np.set_printoptions(threshold=np.inf, linewidth=np.nan)





reflines = np.genfromtxt('reflines_tex.dat',names=True,dtype=None)


# plt.figure(0)
# days = np.arange(0,29)
# print(days)
# cases = np.zeros_like(days)
# cases[:] = np.nan
# cases[0] = 2; cases[7] = 12; cases[14] = 32; cases[15] = 39; cases[16] = 61; cases[17] = 74; cases[19] = 96; cases[20] = 107; cases[21] = 115
# plt.loglog(days, cases, 'ko')
# plt.xlim(0,30)
# plt.ylim(0,130)
# plt.xlabel('days')
# plt.ylabel('cases')
# plt.title('Bolivia COVID-19 cases')
# plt.show()
# print(blah)

###################################
###################################
###################################
###################################
###################################
###################################
###################################


def vac_to_air(wave):
    air = wave / (1.0 + 2.735182E-4 + 131.4182 / (wave**2.) + 2.76249E8 / (wave**4.))
    return air






###################################
###################################
###################################
###################################
###################################
###################################
###################################

    #######################################################################################
###    This function takes an individual galaxy in the lega-c survey and uses ppxf  ###
###    to fit a combination of Conroy+ SSP templates and emission lines to a lega-c ###
###    spectrum.                                                                    ###
###                                                                                 ###
#######################################################################################
##
####################################      Inputs:       ########################################
##
## object_id = the id number of the galaxy in the survey (really just used for catalog output 
##             and figure labels)
##
## z_init = The initial guess for the spectroscopic redshift of the object. This has to be a 
##          a reasonably good guess since pPXF does a poor job of redshift estimation. Not a 
##          problem here since these are known.
##
## specfile = The filename of the spectrum to be fit
## 
## whtfile = The filename of the weightfile to be fit
##
## Optional Inputs:
##
## emsub_specfile = The filename of a fits file to output an emission line-subtracted spectrum
##                  (I've commented this out for you)
##
## plotfile = The filename of an output plot of the best-fit
##
## outfile = An open file object to which to write a line including the best-fit parameters
##
## outfile_spec = An open file object to which to output the log-binned spectrum and best-fit
##                models
##
## mpoly = The order of the multiplicative polynomial (I recommend just using the defaults)
##
## apoly = The order of the additive polynomial (I recommend just using the defaults)
## 
## reflines = A catalog of reference lines to lable on the output plot
##
##
####################################      Outputs:       ########################################
##
## zfit_stars, ezfit_stars = redshift (and error) of the continuum model 
##
## sigma_stars, esigma_stars = sigma (and error) of the continuum model
##
## zfit_gas, ezfit_gas = redshift (and error) of the emission lines
##
## sigma_gas, esigma_gas = sigma (and error) of the emission lines
##
## SN_median = Median Signal-to-noise in the full spectrum
## 
## SN_rf_4000 = Signal-to-noise at rest-frame 4000 Angstroms
## 
## SN_obs_8030 = Signal-to-noise at observed-frame 8030 Angstroms
## 
## chi2 = chi^2 value from PPXF associated with the best-fit model
##
from pydl.goddard.astro import airtovac, vactoair
def ppxf_single(object_id, z_init, lambda_spec, galaxy_lin, error_lin,cars_model, emsub_specfile=None,
              plotfile=None, outfile=None, outfile_spec=None, mpoly=None, apoly=None,
              reflines=None):

    # Speed of light
    c = 299792.458

    
    #h_spec = spec_hdu[0].header
    #Ang_air = h_spec['CRVAL3'] + np.arange(0,h_spec['CDELT3']*(h_spec['NAXIS3']),h_spec['CDELT3'])
    
    #s = 10**4/Ang_air
    #n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.000159740894897 / (38.92568793293 - s**2)
    #lambda_spec = Ang_air*n

    #wave = h_spec['CRVAL3'] + np.arange(0,h_spec['CDELT3']*(h_spec['NAXIS3']),h_spec['CDELT3'])
    
    #lambda_spec = vactoair(wave)
    #lambda_spec = h_spec['CRVAL3'] + np.arange(0,h_spec['CDELT3']*(h_spec['NAXIS3']),h_spec['CDELT3'])
    # Crop to finite
    use = (np.isfinite(galaxy_lin) & (galaxy_lin > 0.0))
    
    

    # Making a "use" vector
    use_indices = np.arange(galaxy_lin.shape[0])[use]
    galaxy_lin = (galaxy_lin[use_indices.min():(use_indices.max()+1)])[2:-3]
    error_lin = (error_lin[use_indices.min():(use_indices.max()+1)])[2:-3]
    lambda_spec = (lambda_spec[use_indices.min():(use_indices.max()+1)])[2:-3]

    
    
    lamRange_gal = np.array([np.min(lambda_spec),np.max(lambda_spec)])

    # New resolution estimate = 0.9 \AA (sigma), converting from sigma to FWHM
    FWHM_gal = 2.355*(np.max(lambda_spec) -  np.min(lambda_spec))/len(lambda_spec)
    print('FWHM', FWHM_gal)
    lamRange_gal = lamRange_gal/(1+float(z_init)) # Compute approximate restframe wavelength range
    FWHM_gal = FWHM_gal/(1+float(z_init))   # Adjust resolution in Angstrom

    sigma_gal = FWHM_gal/(2.3548*4500.0)*c # at ~4500 \AA
    
    # log rebinning for the fits
    galaxy, logLam_gal, velscale = util.log_rebin(np.around(lamRange_gal,decimals=3), galaxy_lin, flux=True)

    noise, logLam_noise, velscale = util.log_rebin(np.around(lamRange_gal,decimals=3), error_lin, \
                                              velscale=velscale, flux=True)
                              

    # correcting for infinite or zero noise
    noise[np.logical_or((noise == 0.0),np.isnan(noise))] = 1.0
    galaxy[np.logical_or((galaxy < 0.0),np.isnan(galaxy))] = 0.0

    if galaxy.shape != noise.shape:
        galaxy = galaxy[:-1]
        logLam_gal = logLam_gal[:-1]
        

    # Define lamRange_temp and logLam_temp
    
    
    
    
    
    
    
    
    
    
    #lamRange_temp, logLam_temp = setup_spectral_library_conroy(velscale[0], FWHM_gal)
    
    # Construct a set of Gaussian emission line templates.
    # Estimate the wavelength fitted range in the rest frame.
    #
    
    gas_templates, line_names, line_wave = util.emission_lines(logLam_gal, lamRange_gal, FWHM_gal)


    
    
    
    
    goodpixels = np.arange(galaxy.shape[0])
    wave = np.exp(logLam_gal)

    # crop very red end (only affects a very small subsample)
    # goodpixels = goodpixels[wave <= 5300]
    
    # exclude lines at the edges (where things can go very very wrong :))
    include_lines = np.where((line_wave > (wave.min()+10.0)) & (line_wave < (wave.max()-10.0)))
    if line_wave[include_lines[0]].shape[0] < line_wave.shape[0]:
        line_wave = line_wave[include_lines[0]]
        line_names = line_names[include_lines[0]]
        gas_templates = gas_templates[:,include_lines[0]]

        
    #reg_dim = stars_templates.shape[1:]
    reg_dim = gas_templates.shape[1:]
    templates =gas_templates
    #np.hstack([gas_templates, gas_templates])

    
    dv = 0#(logLam_temp[0]-logLam_gal[0])*c # km/s

    vel = 0#np.log(z_init + 1)*c # Initial estimate of the galaxy velocity in km/s
    #z = np.exp(vel/c) - 1   # Relation between velocity and redshift in pPXF


    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #    
    t = clock()

    nNLines = gas_templates.shape[1]
    component =  [0]*nNLines
    start_gas = [vel, 100.]
    start = start_gas # adopt the same starting value for both gas (BLs and NLs) and stars
    moments = [2] # fit (V,sig,h3,h4) for the stars and (V,sig) for the gas' broad and narrow components

    fixed = None

    

    ## Additive polynomial degree
    
    if apoly is None: degree = int(np.ceil((lamRange_gal[1]-lamRange_gal[0])/(100.0*(1.+float(z_init)))))
    else: degree = int(apoly)

    if mpoly is None: mdegree = 3
    else: mdegree = int(mpoly)
    
    
    # Trying: sigmas must be kinematically decoupled
    
    # #Trying: velocities must be kinematically decoupled
    #A_ineq = [[0,0,2,0,-1,0]]
    #b_ineq = [0]

    bounds_gas = [[-1000,1000], [0,1000]]
    bounds = bounds_gas
    
    
    
    pp = ppxf.ppxf(templates, galaxy, noise, velscale, start, fixed=fixed,
              plot=False, moments=moments, mdegree=mdegree,
              degree=degree, vsyst=dv, reg_dim = reg_dim,
              goodpixels=goodpixels, bounds=bounds)
    #component=component,
    
    # redshift_to_newtonian:
    # return (v-astropy.constants.c.to('km/s').value*redshift)/(1.0+redshift)
    
    # "v" from above is actually the converted velocity which is
    # (np.exp(v_out/c) - 1)*c
    
    v_gas =pp.sol[0]
    ev_gas = pp.error[0]
    
    
    
    conv_vel_gas = (np.exp(pp.sol[0]/c) - 1)*c
    
    
    vel_gas = (1 + z_init)*(conv_vel_gas - c*z_init)
    
    sigma_gas = pp.sol[1] #first # is template, second # is the moment
    esigma_gas = pp.error[1]#*np.sqrt(pp.chi2)
    
    
    zfit_gas = (z_init + 1)*(1 + pp.sol[0]/c) - 1
    
    zfit_stars = z_init

    ezfit_gas = (z_init + 1)*pp.error[0]*np.sqrt(pp.chi2)/c
   
    
   
    

    if plotfile is not None:
        #
        ### All of the rest of this plots and outputs the results of the fit
        ### Feel free to comment anything out at will
        #

        maskedgalaxy = np.copy(galaxy)
        lowSN = np.where(noise > (0.9*np.max(noise)))
        maskedgalaxy[lowSN] = np.nan

        wave = np.exp(logLam_gal)*(1.+z_init)/(1.+zfit_stars)

        fig = plt.figure(figsize=(12,7))
        ax1 = fig.add_subplot(211)

        # plotting smoothed spectrum
        smoothing_fact = 3

        ax1.plot(wave,convolve(maskedgalaxy, Box1DKernel(smoothing_fact)),color='Gray',linewidth=0.5)
        ax1.plot(wave[goodpixels],convolve(maskedgalaxy[goodpixels], Box1DKernel(smoothing_fact)),'k',linewidth=1.)

        label = "Best fit template from high res Conroy SSPs + emission lines at z={0:.3f}".format(zfit_stars)

        # overplot stellar templates alone
        ax1.plot(wave, pp.bestfit, 'r', linewidth=1.0,alpha=0.75, label=label) 

        ax1.set_ylabel('Flux')
        ax1.legend(loc='upper right',fontsize=10)
        ax1.set_title(object_id)

        xmin, xmax = ax1.get_xlim()

        ax2 = fig.add_subplot(413, sharex=ax1, sharey=ax1)

        # plotting emission lines if included in the fit
        
        
        gas = pp.matrix[:,-nNLines:].dot(pp.weights[-nNLines:])
        
        ax2.plot(wave, gas, 'b', linewidth=2,\
                 label = '$\sigma_{gas}$'+'={0:.0f}$\pm${1:.0f} km/s'.format(sigma_gas, esigma_gas)+', $V_{gas}$'+'={0:.0f}$\pm${1:.0f} km/s'.format(v_gas, ev_gas)) # overplot emission lines alone
        cars_model = (cars_model[use_indices.min():(use_indices.max()+1)])[2:-3]
        ax2.plot(np.array(lambda_spec)/(1+z_init), cars_model, color='orange', linewidth=1,\
            label = 'CARS Model')
        
        #(lambda_spec[use_indices.min():(use_indices.max()+1)])[2:-3]
        stars = pp.bestfit - gas
        

        
        
        #if (ymax > 3.0*np.median(stars)): ymax = 3.0*np.median(stars)
        #if (ymin < -0.5): ymin = -0.5

        

        ax2.set_ylabel('Best Fits')

        ax2.legend(loc='upper left',fontsize=10)

        # Plotting the residuals
        ax3 = fig.add_subplot(817, sharex=ax1)
        ax3.plot(wave[goodpixels], (convolve(maskedgalaxy, Box1DKernel(smoothing_fact))-pp.bestfit)[goodpixels], 'k',label='Fit Residuals')
        #ax3.set_yticks([-0.5,0,0.5])
        ax3.set_ylabel('Residuals')

        ax4 = fig.add_subplot(818, sharex=ax1)

        ax4.plot(wave, noise, 'k',label='Flux Error')

        ax4.set_ylabel('Noise')
        ax4.set_xlabel('Rest Frame Wavelength [$\AA$]')
        #ax4.set_yticks(np.arange(0,0.5,0.1))

        '''if reflines is not None:
            for i,w,label in zip(range(len(reflines)),reflines['wave'],reflines['label']):
                if ((w > xmin) and (w < xmax)):
#                     ax1.text(w,ymin+(ymax-ymin)*(0.03+0.08*(i % 2)),'$\mathrm{'+label+'}$',fontsize=10,\
#                              horizontalalignment='center',\
#                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    print(label.decode("utf-8"))
                    ax1.text(w,ymin+(ymax-ymin)*(0.03+0.08*(i % 2)),'$\mathrm{'+label.decode("utf-8")+'}$',fontsize=10,\
                             horizontalalignment='center',\
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    ax1.plot([w,w],[ymin,ymax],':k',alpha=0.5)'''

        fig.subplots_adjust(hspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        #print('Saving figure to {0}'.format(plotfile))
        
        ax1.set_xlim([6450,6750])

        ax2.set_xlim([6450,6750])
        
        #ymin, ymax = ax1.get_ylim([])
        
        
        
        plt.savefig(plotfile,dpi=150)
        plt.close()
    return v_gas,vel_gas, sigma_gas, pp.chi2
    #    print('# id z_stars ez_stars sigma_stars esigma_stars z_gas ez_gas sigma_gas esigma_gas SN_median SN_rf_4000 SN_obs_8030 chi2dof')
    print('{0:d} {1:.6f} {2:.6f} {3:.1f} {4:.1f} {5:.6f} {6:.6f} {7:.1f} {8:.1f} {9:.1f} {10:.1f} {11:.1f} {12:.2f}\n'.format(\
                      object_id,zfit_stars,ezfit_stars, sigma_stars,esigma_stars,\
                      zfit_gas, ezfit_gas, sigma_gas, esigma_gas, SN_median, SN_rf_4000, SN_obs_8030, pp.chi2))
    
    ## This prints the fit parameters to an open file object called outfile
    if outfile is not None:
        outfile.write('{0:d} {1:d} {2:.6f} {3:.6f} {4:.1f} {5:.1f} {6:.6f} {7:.6f} {8:.1f} {9:.1f} {10:.1f} {11:.1f} {12:.1f} {13:.2f}\n'.format(\
                      object_id, row2D, zfit_stars, ezfit_stars, sigma_stars,esigma_stars,\
                      zfit_gas, ezfit_gas, sigma_gas, esigma_gas, SN_median, SN_rf_4000, SN_obs_8030, pp.chi2))

    ## This outputs the spectrum and best-fit model to an open file object called outfile_spec
    if outfile_spec is not None:
        outfile_spec.write('# l f ef f_stars f_gas f_model_tot used_in_fit add_poly mult_poly\n')
        for i in np.arange(wave.shape[0]):
            isgood = 0
            if goodpixels[goodpixels == i].shape[0] == 1: isgood = 1
            outfile_spec.write('{0:0.4f} {1:0.4f} {2:0.4f} {3:0.4f} {4:0.4f} {5:0.4f} {6} {7:0.4f} {8:0.4f}\n'.format(wave[i],galaxy[i],noise[i],stars[i],gas[i],pp.bestfit[i],isgood,add_polynomial[i],mult_polynomial[i]))
            
#     ## This outputs the best-fit emission-subtracted spectrum to a fits file
#     ## but I've commented it out since you are unlikely to need this!
#     if emsub_specfile is not None:
#         wave = np.exp(logLam_gal)*(1.+z_init)
#         if include_gas:
#             emsub = galaxy - gas
#         else:
#             emsub = galaxy
        
#         col1 = fits.Column(name='wavelength', format='E', array=wave)
#         col2 = fits.Column(name='flux',format='E',array=galaxy)
#         col3 = fits.Column(name='error',format='E',array=noise)
#         col4 = fits.Column(name='flux_emsub',format='E',array=emsub)
        
#         cols = fits.ColDefs([col1,col2,col3,col4])
        
#         tbhdu = fits.BinTableHDU.from_columns(cols)
        
#         # delete old file if it exists
#         if os.path.isfile(emsub_specfile): os.remove(emsub_specfile)
#         tbhdu.writeto(emsub_specfile)

    # return zfit_stars, ezfit_stars, sigma_stars, esigma_stars, zfit_gas, ezfit_gas, \
    #     sigma_gas, esigma_gas, SN_median, SN_rf_4000, SN_obs_8030, pp.chi2
    return zfit_stars, sigma_stars, sigma_blr, wave, pp.bestfit



###################################
###################################
###################################
###################################
###################################
###################################
###################################
# running ppxf on interesting spectra:

# ppxf_indiv(138986,0.694,specfile='138986/legac_M1_v3.7_spec1d_138986.fits',\
#           whtfile='138986/legac_M1_v3.7_wht1d_138986.fits',\
#           plotfile='M1_138986_dg.pdf',reflines=reflines)

# ppxf_indiv(138986,0.694,specfile='138986/legac_M2_v3.7_spec1d_138986.fits',\
#           whtfile='138986/legac_M2_v3.7_wht1d_138986.fits',\
#           plotfile='M2_138986_dg.pdf',reflines=reflines)

# ppxf_indiv(132979,0.728,specfile='132979/legac_M1_v3.7_spec1d_132979.fits',\
#           whtfile='132979/legac_M1_v3.7_wht1d_132979.fits',\
#           plotfile='M1_132979_dg.pdf',reflines=reflines)

# ppxf_indiv(132979,0.728,specfile='132979/legac_M101_v3.7_spec1d_132979.fits',\
#           whtfile='132979/legac_M101_v3.7_wht1d_132979.fits',\
#           plotfile='M101_132979_dg.pdf',reflines=reflines)

# ppxf_indiv(143863,0.896,specfile='143863/legac_M8_v3.7_spec1d_143863.fits',\
#           whtfile='143863/legac_M8_v3.7_wht1d_143863.fits',\
#           plotfile='M8_143863_dg.pdf',reflines=reflines)

# ppxf_indiv(123630,0.894,specfile='123630/legac_M4_v3.7_spec1d_123630.fits',\
#           whtfile='123630/legac_M4_v3.7_wht1d_123630.fits',\
#           plotfile='M4_123630_dg.pdf',reflines=reflines)

# ppxf_indiv(123424,0.601,specfile='123424/legac_M7_v3.7_spec1d_123424.fits',\
#           whtfile='123424/legac_M7_v3.7_wht1d_123424.fits',\
#           plotfile='M7_123424_dg.pdf',reflines=reflines)

def closest_5100(lst): 
    return min(range(len(lst)), key = lambda i: abs(lst[i]-5100))

def BH_mass_Hbeta(sigmaHbeta, lmbda, L5100):
    f = 4.47
    MBH = f * (10**6.819) * (sigmaHbeta/1000)**2.0 * (lmbda*L5100/(10**44))**0.533
    return MBH

# M1_132979
# gal1 = ppxf_indiv(132979,0.728,specfile='../LEGAC_v3.7_1D_spectra/legac_M1_v3.7_spec1d_132979.fits',\
#           whtfile='../LEGAC_v3.7_1D_spectra/legac_M1_v3.7_wht1d_132979.fits',\
#           plotfile='../BLR_DGfitting/M1_132979_dg.pdf',reflines=reflines)

# cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)
# z_gal1 = gal1[0]
# d_L = (cosmo.luminosity_distance(z_gal1)).to(u.cm).value
# sigma_blr_gal1 = gal1[2]
# sigma_stars_gal1 = gal1[1]
# lambda_rf_gal1 = gal1[3]
# flux_gal1 = gal1[5]
# F5100_gal1 = flux_gal1[closest_5100(lambda_rf_gal1)] * 10**(-19)
# L5100_gal1 = F5100_gal1 * 4.0 * np.pi * (d_L**2)
# lambda_5100_gal1 = 5100.0*(1.0+z_gal1)
# Mbh_gal1 = BH_mass_Hbeta(sigma_blr_gal1, lambda_5100_gal1, L5100_gal1)
# print(Mbh_gal1)
# print(blah)


# M1_138986
#ppxf_indiv(HE0045,0.694,specfile='../LEGAC_v3.7_1D_spectra/legac_M1_v3.7_spec1d_138986.fits',\
#          whtfile='../LEGAC_v3.7_1D_spectra/legac_M1_v3.7_wht1d_138986.fits',\
#          plotfile='../BLR_DGfitting/M1_138986_dg.pdf',reflines=reflines)
