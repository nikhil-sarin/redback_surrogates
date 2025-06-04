from collections import namedtuple
from redback_surrogates.utils import citation_wrapper
from joblib import load
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import astropy.units as uu
from functools import lru_cache
import os
dirname = os.path.dirname(__file__)
data_folder = os.path.join(dirname, "surrogate_data")

@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
class EnhancedSpectralModel:
    def __init__(self, latent_dim=64, learning_rate=1e-3, use_pca=True, pca_components=32):
        """Initialize the enhanced spectral model with optimized parameters

        Args:
            latent_dim: Dimension of latent space (reduced from 256 to 64)
            learning_rate: Learning rate for model training
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components if use_pca is True
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.encoder = None
        self.decoder = None
        self.regressor = None
        self.param_scaler = None
        self.flux_scaler = None
        self.latent_scaler = None
        self.standard_times = None
        self.standard_freqs = None
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.pca = None

    def predict_spectrum(self, parameters):
        """Predict spectrum for given parameters

        Args:
            parameters: DataFrame or array of physical parameters

        Returns:
            Predicted spectrum (time_dim, freq_dim)
        """
        # Convert to numpy array if DataFrame
        if isinstance(parameters, pd.DataFrame):
            param_array = parameters.values
        else:
            param_array = np.atleast_2d(parameters)

        # Scale parameters
        param_scaled = self.param_scaler.transform(param_array)

        # Predict latent representation (scaled)
        scaled_latent = self.regressor.predict(param_scaled, verbose=0)

        if self.use_pca and self.pca is not None:
            # Inverse scale the reduced latent space
            reduced_latent = self.inverse_scale_latent(scaled_latent)

            # Inverse transform to full latent space
            latent = self.pca.inverse_transform(reduced_latent)
        else:
            # Direct inverse scaling of latent space
            latent = self.inverse_scale_latent(scaled_latent)

        # Decode to scaled spectrum
        scaled_spectrum = self.decoder.predict(latent, verbose=0)

        # Inverse scale to original flux range
        spectrum = self.inverse_preprocess_flux(scaled_spectrum)

        # Return first spectrum if only one set of parameters
        if param_array.shape[0] == 1:
            return spectrum[0]

        return spectrum

    def inverse_preprocess_flux(self, scaled_flux):
        """Convert scaled flux back to original scale

        Args:
            scaled_flux: Scaled flux arrays

        Returns:
            Original scale flux arrays
        """
        if self.flux_scaler is None:
            print("Warning: flux_scaler not found, returning unscaled data")
            return scaled_flux

        # Reshape to 2D for inverse scaling
        orig_shape = scaled_flux.shape
        flux_2d = scaled_flux.reshape(orig_shape[0], -1)

        # Apply inverse scaling
        orig_2d = self.flux_scaler.inverse_transform(flux_2d)

        # Reshape back to original shape
        orig_flux = orig_2d.reshape(orig_shape)

        return orig_flux

    def inverse_scale_latent(self, scaled_latent):
        """Convert scaled latent values back to original scale

        Args:
            scaled_latent: Scaled latent values

        Returns:
            Original scale latent values
        """
        if self.latent_scaler is None:
            print("Warning: latent_scaler not found, returning unscaled data")
            return scaled_latent

        # Apply inverse scaling
        original_latent = self.latent_scaler.inverse_transform(scaled_latent)

        return original_latent

    @classmethod
    def load_model(cls, directory=data_folder + '/TypeII_Moriya'):
        """Load saved model from disk

        Args:
            directory: Directory containing saved model

        Returns:
            EnhancedSpectralModel instance with loaded models
        """
        # Load configuration
        import json
        with open(os.path.join(directory, 'config.json'), 'r') as f:
            config = json.load(f)

        # Initialize model with loaded config
        model = cls(
            latent_dim=config['latent_dim'],
            use_pca=config['use_pca'],
            pca_components=config['pca_components']
        )

        # Load encoder and decoder
        model.encoder = tf.keras.models.load_model(os.path.join(directory, 'encoder.keras'))
        model.decoder = tf.keras.models.load_model(os.path.join(directory, 'decoder.keras'))

        # Load regressor
        model.regressor = tf.keras.models.load_model(os.path.join(directory, 'regressor.keras'))

        # Load scalers
        import joblib
        model.param_scaler = joblib.load(os.path.join(directory, 'param_scaler.pkl'))
        model.flux_scaler = joblib.load(os.path.join(directory, 'flux_scaler.pkl'))

        # Load latent scaler if exists
        latent_scaler_path = os.path.join(directory, 'latent_scaler.pkl')
        if os.path.exists(latent_scaler_path):
            model.latent_scaler = joblib.load(latent_scaler_path)

        # Load PCA if exists
        pca_path = os.path.join(directory, 'pca.pkl')
        if os.path.exists(pca_path):
            model.pca = joblib.load(pca_path)

        # Load grid information
        grids = np.load(os.path.join(directory, 'standard_grids.npz'))
        model.standard_times = grids['times']
        model.standard_freqs = grids['freqs']

        return model

# Cached model loaders
@lru_cache(maxsize=1)
def _load_lbol_models():
    """Load and cache the lbol models and scalers."""
    lbolscaler = load(data_folder + '/TypeII_Moriya/lbolscaler.save')
    lbol_model = keras.saving.load_model(data_folder + '/TypeII_Moriya/lbol_model.keras')
    xscaler = load(data_folder + '/TypeII_Moriya/xscaler.save')
    return lbolscaler, lbol_model, xscaler

@lru_cache(maxsize=1)
def _load_photosphere_models():
    """Load and cache the photosphere models and scalers."""
    xscaler = load(data_folder + '/TypeII_Moriya/xscaler.save')
    tempscaler = load(data_folder + '/TypeII_Moriya/temperature_scaler.save')
    radscaler = load(data_folder + '/TypeII_Moriya/radius_scaler.save')
    temp_model = keras.saving.load_model(data_folder + '/TypeII_Moriya/temp_model.keras')
    rad_model = keras.saving.load_model(data_folder + '/TypeII_Moriya/radius_model.keras')
    return xscaler, tempscaler, radscaler, temp_model, rad_model

@lru_cache(maxsize=1)
def _load_spectra_model():
    """Load and cache the spectra model."""
    return EnhancedSpectralModel.load_model()

# Optional: Function to clear all caches if needed
def clear_typeII_model_cache():
    """Clear all cached models to free memory."""
    _load_lbol_models.cache_clear()
    _load_photosphere_models.cache_clear()
    _load_spectra_model.cache_clear()


# Updated functions using cached models
@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_lbol(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Generate bolometric light curve for Type II supernovae based on physical parameters (vectorised)

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: tts (in days in source frame) and bolometric luminosity (in erg/s)
    """
    rcsm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)

    # Load cached models
    lbolscaler, lbol_model, xscaler = _load_lbol_models()

    tts = np.geomspace(1e-1, 400, 200)
    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    ss = xscaler.transform(ss)
    lbols = lbol_model(ss)
    lbols = lbolscaler.inverse_transform(lbols)
    if isinstance(progenitor, float):
        lbols = lbols.flatten()
    return tts, 10 ** lbols


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_photosphere(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Generate synthetic photospheric temperature and radius for Type II supernovae based on physical parameters.
    (vectorised)

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: tts (in days in source frame) and temp (in K) and radius (in cm)
    """
    rcsm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)

    # Load cached models
    xscaler, tempscaler, radscaler, temp_model, rad_model = _load_photosphere_models()

    tts = np.geomspace(1e-1, 400, 200)
    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    ss = xscaler.transform(ss)
    temp = temp_model(ss)
    rad = rad_model(ss)
    temp = tempscaler.inverse_transform(temp)
    rad = radscaler.inverse_transform(rad)
    if isinstance(progenitor, float):
        temp = temp.flatten()
        rad = rad.flatten()
    return tts, temp, rad


@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025arXiv250602107S/abstract, https://ui.adsabs.harvard.edu/abs/2023PASJ...75..634M/abstract")
def typeII_spectra(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """
    Generate synthetic spectra for Type II supernovae based on physical parameters.

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param log10_mdot: in solar masses per year
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs: None
    :return: rest-frame spectrum (luminosity density), frequency (in Angstrom) and time arrays and times (in days in source frame)
    """
    rcsm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)
    # Create standard grids
    standard_times = np.geomspace(0.1, 400, 50)
    standard_freqs = np.geomspace(500, 49500, 50)

    # Create parameter dataframe for the surrogate model
    new_params = pd.DataFrame([{
        'progenitor': progenitor,
        'ni_mass': ni_mass,
        'mass_loss': log10_mdot,
        'beta': beta,
        'csm_radius': rcsm,
        'explosion_energy': esn
    }])

    # Get cached model
    model = _load_spectra_model()

    predicted_spectrum = model.predict_spectrum(new_params)
    # Apply empirical correction factor (if needed)
    predicted_spectrum = 10 ** predicted_spectrum
    # Convert to physical units (erg/s/Hz)
    rest_spectrum = predicted_spectrum * uu.erg / uu.s / uu.Hz

    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=rest_spectrum,
        frequency=standard_freqs * uu.Angstrom,
        time=standard_times * uu.day
    )
