import numpy as np
import kilonovanet as knnet
from collections import namedtuple
from redback_surrogates.utils import citation_wrapper, convert_to_observer_frame
import os
dirname = os.path.dirname(__file__)

@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1137L/abstract')
def bulla_bns_kilonovanet_spectra(time_source_frame, redshift, mej_dyn, mej_disk, phi, costheta_obs, **kwargs):
    """
    Kilonovanet model based on Bulla BNS merger simulations

    :param time_source_frame: time in source frame in days
    :param redshift: redshift
    :param mej_dyn: dynamical mass of ejecta in solar masses
    :param mej_disk: disk mass of ejecta in solar masses
    :param phi: half-opening angle of the lanthanide-rich tidal dynamical ejecta in degrees
    :param costheta_obs: cosine of the observers viewing angle
    :return: named tuple with observer frame time in days, observer frame wavelength in Angstroms,
    and spectra in observer frame in erg/s/Angstrom
    """
    metadata_file = f"{dirname}/surrogate_data/metadata_bulla_bns.json"
    torch_file = f"{dirname}/surrogate_data/bulla-bns-latent-20-hidden-1000-CV-4.pt"
    wavelength = np.linspace(100.0, 99900, 500)
    time_observer_frame, wavelength_observer_frame = convert_to_observer_frame(time_source_frame, wavelength, redshift)
    physical_parameters = np.array([mej_dyn, mej_disk, phi, costheta_obs])
    model = knnet.Model(metadata_file, torch_file)
    spectra, unique_t = model.predict_spectra(physical_parameters, time_source_frame)
    return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                  lambdas=wavelength_observer_frame,
                                                                  spectra=spectra)
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1137L/abstract')
def bulla_nsbh_kilonovanet_spectra(time_source_frame, redshift, mej_dyn, mej_disk, costheta_obs, **kwargs):
    """
    Kilonovanet model based on Bulla NSBH merger simulations

    :param time_source_frame: time in source frame in days
    :param redshift: redshift
    :param mej_dyn: dynamical mass of ejecta in solar masses
    :param mej_disk: disk mass of ejecta in solar masses
    :param costheta_obs: cosine of the observers viewing angle
    :return: named tuple with observer frame time in days, observer frame wavelength in Angstroms,
    and spectra in observer frame in erg/s/Angstrom
    """
    metadata_file = f"{dirname}/surrogate_data/metadata_bulla_bhns.json"
    torch_file = f"{dirname}/surrogate_data/bulla-bhns-latent-2-hidden-500-CV-4.pt"
    wavelength = np.linspace(100.0, 99900, 500)
    time_observer_frame, wavelength_observer_frame = convert_to_observer_frame(time_source_frame, wavelength, redshift)
    physical_parameters = np.array([mej_dyn, mej_disk, costheta_obs])
    model = knnet.Model(metadata_file, torch_file)
    spectra, unique_t = model.predict_spectra(physical_parameters, time_source_frame)
    return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                  lambdas=wavelength_observer_frame,
                                                                  spectra=spectra)
@citation_wrapper('https://ui.adsabs.harvard.edu/abs/2022MNRAS.516.1137L/abstract')
def kasen_bns_kilonovanet_spectra(time_source_frame, redshift, mej, vej, chi, **kwargs):
    """
    Kilonovanet model based on Kasen BNS merger simulations

    :param time_source_frame: time in source frame in days
    :param redshift: redshift
    :param mej: ejecta mass in solar masses
    :param vej: ejecta velocity in units of c
    :param chi: lanthanide fraction
    :return: named tuple with observer frame time in days, observer frame wavelength in Angstroms,
    and spectra in observer frame in erg/s/Angstrom
    """
    metadata_file = f"{dirname}/surrogate_data/metadata_kasen_bns.json"
    torch_file = f"{dirname}/surrogate_data/kasen-bns-latent-10-hidden-500-CV-4.pt"
    wavelength = 10 ** (2.175198139181011 + np.linspace(0, 1, 1629) * 2.8224838828121763)
    time_observer_frame, wavelength_observer_frame = convert_to_observer_frame(time_source_frame, wavelength, redshift)
    physical_parameters = np.array([mej, vej, chi])
    model = knnet.Model(metadata_file, torch_file)
    spectra, unique_t = model.predict_spectra(physical_parameters, time_source_frame)
    return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                  lambdas=wavelength_observer_frame,
                                                                  spectra=spectra)