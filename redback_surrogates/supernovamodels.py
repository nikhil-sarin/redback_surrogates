from redback_surrogates.utils import citation_wrapper
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import os
dirname = os.path.dirname(__file__)
import numpy as np
import pandas as pd

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

    @classmethod
    def load_model(cls, directory='enhanced_spectral_model'):
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


def lbol_vec(progenitor, ni_mass, mdot, beta, rcsm, esn, **kwargs):
    """

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param mdot: in 10^{-x} solar masses per year so x is the number
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs:
    :return: tts (in days in source frame) and bolometric luminosity (in erg/s)
    """
    rcsm = rcsm * 1e14
    lbolscaler = load('May2025_processed_data/lbolscaler.save')
    lbol_model = keras.saving.load_model('May2025_processed_data/lbol_model.keras')
    tts = np.geomspace(1e-1, 400, 200)
    ss = np.array([progenitor, ni_mass, mdot, beta, rcsm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    ss = xscaler.transform(ss)
    lbols = lbol_model(ss)
    lbols = lbolscaler.inverse_transform(lbols)
    if isinstance(progenitor, float):
        lbols = lbols.flatten()
    return tts, 10**lbols

def photosphere_vec(progenitor, ni_mass, mdot, beta, rcsm, esn, **kwargs):
    """

    :param progenitor: in solar masses
    :param ni_mass: in solar masses
    :param mdot: in 10^{-x} solar masses per year so x is the number
    :param beta: dimensionless
    :param rcsm: in 10^14 cm
    :param esn: in 10^51
    :param kwargs:
    :return: tts (in days in source frame) and temp (in K) and radius (in cm)
    """
    rcsm = rcsm * 1e14
    tempscaler = load('May2025_processed_data/temperature_scaler.save')
    radscaler = load("May2025_processed_data/radius_scaler.save")
    temp_model = keras.saving.load_model('May2025_processed_data/temp_model.keras')
    rad_model = keras.saving.load_model('May2025_processed_data/radius_model.keras')
    tts = np.geomspace(1e-1, 400, 200)
    ss = np.array([progenitor, ni_mass, mdot, beta, rcsm, esn]).T
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