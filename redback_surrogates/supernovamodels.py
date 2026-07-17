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
import warnings
import copy
import h5py
import torch
import torch.nn as nn
dirname = os.path.dirname(__file__)
data_folder = os.path.join(dirname, "surrogate_data")


def _resolve_torch_device(device=None):
    """
    Resolve torch execution device.

    Default behavior is GPU-first (CUDA, then MPS) with CPU fallback.
    """
    if isinstance(device, torch.device):
        requested = str(device)
    elif device is None:
        requested = "auto"
    else:
        requested = str(device).strip().lower()

    if requested in ("", "none", "auto", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        resolved = torch.device(requested)
    except (TypeError, RuntimeError, ValueError):
        warnings.warn(
            f"Unknown torch device '{device}', falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    if resolved.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"CUDA device '{device}' requested but CUDA is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        warnings.warn(
            f"MPS device '{device}' requested but MPS is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return resolved


def _canonical_device_key(device):
    """Return a stable device key for cache indexing."""
    dev = _resolve_torch_device(device)
    if dev.type == "cuda":
        idx = dev.index
        if idx is None:
            idx = torch.cuda.current_device()
        return f"cuda:{idx}"
    return dev.type


def _to_numpy_array(x):
    """Convert tensor/array-like to numpy array on CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

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


# ========== Interaction-Model ==========

class _InteractionResBlock(nn.Module):
    """Pre-activation ResNet block with SiLU for Interaction-Model."""
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = torch.nn.functional.silu(self.norm1(x))
        h = self.fc1(h)
        h = torch.nn.functional.silu(self.norm2(h))
        h = self.fc2(h)
        return x + h


class _InteractionPhysicsEmbedding(nn.Module):
    """Physics parameter embedding for Interaction-Model."""
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feats = [x, x ** 2, torch.tanh(x), torch.log(torch.abs(x) + 1.1)]
        if x.shape[1] >= 2:
            feats.append(x[:, 0:1] * x[:, 1:2])
        return torch.cat(feats, dim=1)


class InteractionLatentEmulator(nn.Module):
    """Interaction-Model latent emulator."""
    def __init__(self, input_dim, latent_dim, hidden_dim=1024, num_blocks=12):
        super().__init__()
        self.register_buffer('x_mean', torch.zeros(input_dim))
        self.register_buffer('x_std', torch.ones(input_dim))
        self.physics_embed = _InteractionPhysicsEmbedding()
        embed_dim = input_dim * 4 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([_InteractionResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        x = self.physics_embed(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class InteractionResNetDecoder(nn.Module):
    """Interaction-Model decoder."""
    def __init__(self, latent_dim, hidden_dim, output_dim, num_blocks):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList([_InteractionResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        return self.output_proj(h)


def _load_interaction_input_stats(bundle, input_dim):
    """Return interaction-model input normalization statistics."""
    if 'X_mean' in bundle and 'X_std' in bundle:
        return bundle['X_mean'], bundle['X_std']

    h5_path = os.environ.get('STELLA_INTERACTION_STATS_H5')
    if h5_path is None:
        raise FileNotFoundError(
            "Interaction-model checkpoint is missing 'X_mean'/'X_std'. "
            "Set STELLA_INTERACTION_STATS_H5 to an HDF5 file containing these datasets."
        )
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"STELLA_INTERACTION_STATS_H5 points to a missing file: '{h5_path}'."
        )

    with h5py.File(h5_path, 'r') as f:
        return f['X_mean'][:input_dim], f['X_std'][:input_dim]


@lru_cache(maxsize=1)
def _load_typeII_interaction_model_torch_bundle():
    """Load and cache the Interaction-Model torch surrogate."""
    directory = os.environ.get(
        'STELLA_INTERACTION_MODEL_DIR',
        os.path.join(data_folder, 'TypeII_Moriya', 'interaction_model'),
    )
    ckpt_path = os.path.join(directory, 'emulator_6param_timeweighted_best.pt')
    bundle = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    cfg = bundle.get('config', {})
    ae_cfg = bundle.get('ae_config', {})
    input_dim = bundle['input_dim']
    latent_dim = bundle['latent_dim']
    flux_min = bundle['flux_min']
    flux_max = bundle['flux_max']
    z_mean = bundle['z_mean']
    z_std = bundle['z_std']
    time_grid = bundle['time_grid']
    wave_grid = bundle['wave_grid']
    n_time = bundle['n_time']
    n_wave = bundle['n_wave']
    X_mean, X_std = _load_interaction_input_stats(bundle, input_dim)

    emulator = InteractionLatentEmulator(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=cfg.get('hidden_dim', 1024),
        num_blocks=cfg.get('num_blocks', 12),
    )
    emulator.load_state_dict(bundle['emulator_state_dict'])
    emulator.eval()

    decoder = InteractionResNetDecoder(
        latent_dim=latent_dim,
        hidden_dim=ae_cfg.get('hidden_dim', 1024),
        output_dim=ae_cfg.get('input_dim', 10000),
        num_blocks=ae_cfg.get('num_blocks', 12),
    )
    decoder.load_state_dict(bundle['decoder_state_dict'])
    decoder.eval()

    return emulator, decoder, z_mean, z_std, flux_min, flux_max, time_grid, wave_grid, n_time, n_wave, X_mean, X_std


@lru_cache(maxsize=4)
def _load_typeII_interaction_model_torch_bundle_on_device(device_key):
    """
    Load Interaction-Model bundle on a specific device once and reuse it.
    """
    emulator_cpu, decoder_cpu, z_mean_cpu, z_std_cpu, flux_min, flux_max, time_grid, wave_grid, n_time, n_wave, X_mean, X_std = \
        _load_typeII_interaction_model_torch_bundle()
    device = torch.device(device_key)

    emulator = copy.deepcopy(emulator_cpu).to(device)
    emulator.eval()
    decoder = copy.deepcopy(decoder_cpu).to(device)
    decoder.eval()
    z_mean = z_mean_cpu.to(device) if torch.is_tensor(z_mean_cpu) else torch.as_tensor(z_mean_cpu, device=device)
    z_std = z_std_cpu.to(device) if torch.is_tensor(z_std_cpu) else torch.as_tensor(z_std_cpu, device=device)
    x_mean_np = _to_numpy_array(X_mean)
    x_std_np = _to_numpy_array(X_std)

    return emulator, decoder, z_mean, z_std, flux_min, flux_max, time_grid, wave_grid, n_time, n_wave, x_mean_np, x_std_np


def clear_typeII_interaction_model_cache():
    """Clear Interaction-Model cache to free memory."""
    _load_typeII_interaction_model_torch_bundle.cache_clear()
    _load_typeII_interaction_model_torch_bundle_on_device.cache_clear()



def typeII_spectra_interaction_model(progenitor, ni_mass, log10_mdot, beta, rcsm, esn, **kwargs):
    """Type II spectra using the Interaction-Model surrogate."""
    rcsm_cm = rcsm * 1e14
    log10_mdot = np.abs(log10_mdot)
    device_key = _canonical_device_key(kwargs.get('device', 'auto'))
    device = torch.device(device_key)

    emulator, decoder, z_mean, z_std, flux_min, flux_max, time_grid, wave_grid, n_time, n_wave, X_mean, X_std = \
        _load_typeII_interaction_model_torch_bundle_on_device(device_key)

    ss = np.array([progenitor, ni_mass, log10_mdot, beta, rcsm_cm, esn]).T
    if isinstance(progenitor, float):
        ss = ss.reshape(1, -1)
    ss = ((ss - X_mean) / X_std).astype(np.float32)

    with torch.no_grad():
        x_tensor = torch.from_numpy(ss).to(device)
        z_norm = emulator(x_tensor)
        z = z_norm * z_std + z_mean
        pred_norm = decoder(z).cpu().numpy()

    pred_logf = pred_norm * (flux_max - flux_min) + flux_min
    pred_logf = pred_logf.reshape(pred_logf.shape[0], n_time, n_wave)
    if isinstance(progenitor, float):
        pred_logf = pred_logf[0]

    pred_flux = (10 ** pred_logf) * uu.erg / uu.s / uu.Hz
    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=pred_flux,
        frequency=np.array(wave_grid) * uu.Angstrom,
        time=np.array(time_grid) * uu.day,
    )


# ========== Photospheric-Model ==========

class _PhotosphericModelCNN2DEncoder(nn.Module):
    """Photospheric-Model CNN encoder: 100x100 to latent."""
    def __init__(self, latent_dim=256, base_ch=32, bottleneck_size=13):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.SiLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.SiLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.SiLU())
        self.fc = nn.Linear(base_ch * 4 * bottleneck_size * bottleneck_size, latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.fc(x.view(x.size(0), -1))


class _PhotosphericModelCNN2DDecoder(nn.Module):
    """Photospheric-Model CNN decoder: latent to 100x100."""
    def __init__(self, latent_dim=256, base_ch=32, bottleneck_size=13):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.bottleneck_channels = base_ch * 4
        self.fc = nn.Linear(latent_dim, self.bottleneck_channels * bottleneck_size * bottleneck_size)
        self.up1 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.SiLU())
        self.up2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.SiLU())
        self.up3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU())
        self.up4 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU())
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(-1, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)
        x = self.up1(x)
        x = nn.functional.interpolate(x, size=(25, 25), mode='bilinear', align_corners=False)
        x = self.up2(x)
        x = nn.functional.interpolate(x, size=(50, 50), mode='bilinear', align_corners=False)
        x = self.up3(x)
        x = nn.functional.interpolate(x, size=(100, 100), mode='bilinear', align_corners=False)
        x = self.up4(x)
        return self.out_conv(x)


class _PhotosphericModelLatentEmulator(nn.Module):
    """Photospheric-Model latent emulator."""
    def __init__(self, input_dim, latent_dim, hidden_dim=512, num_blocks=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


@lru_cache(maxsize=1)
def _load_photospheric_model_cnn_v3():
    """Load and cache the Photospheric-Model surrogate."""
    directory = os.environ.get(
        'STELLA_PHOTOSPHERIC_MODEL_DIR',
        os.path.join(data_folder, 'TypeII_Moriya', 'photospheric_model'),
    )
    ae_path = os.path.join(directory, 'ae_cnn_v3_best.pt')
    emulator_path = os.path.join(directory, 'emulator_cnn_v3_6param_best.pt')
    ae_ckpt = torch.load(ae_path, map_location='cpu', weights_only=False)
    em_ckpt = torch.load(emulator_path, map_location='cpu', weights_only=False)

    latent_dim = ae_ckpt['latent_dim']
    base_ch = ae_ckpt['base_channels']
    bottleneck_size = ae_ckpt['bottleneck_size']
    Y_min, Y_max = ae_ckpt['Y_min'], ae_ckpt['Y_max']
    wave_grid, time_grid = ae_ckpt['wave_grid'], ae_ckpt['time_grid']

    decoder = _PhotosphericModelCNN2DDecoder(latent_dim, base_ch, bottleneck_size)
    decoder_state = {
        k.replace('decoder.', ''): v
        for k, v in ae_ckpt['model_state_dict'].items()
        if k.startswith('decoder.')
    }
    decoder.load_state_dict(decoder_state)
    decoder.eval()

    input_dim = em_ckpt['input_dim']
    hidden_dim = em_ckpt['hidden_dim']
    num_blocks = em_ckpt['num_blocks']
    z_mean, z_std = em_ckpt['z_mean'], em_ckpt['z_std']
    X_mean, X_std = em_ckpt['X_mean'], em_ckpt['X_std']

    emulator = _PhotosphericModelLatentEmulator(input_dim, latent_dim, hidden_dim, num_blocks)
    emulator.load_state_dict(em_ckpt['emulator_state_dict'])
    emulator.eval()
    return emulator, decoder, z_mean, z_std, Y_min, Y_max, wave_grid, time_grid, X_mean, X_std


@lru_cache(maxsize=4)
def _load_photospheric_model_cnn_v3_on_device(device_key):
    """Load Photospheric-Model bundle on a specific device once and reuse it."""
    emulator_cpu, decoder_cpu, z_mean_cpu, z_std_cpu, Y_min, Y_max, wave_grid, time_grid, X_mean, X_std = \
        _load_photospheric_model_cnn_v3()
    device = torch.device(device_key)

    emulator = copy.deepcopy(emulator_cpu).to(device)
    emulator.eval()
    decoder = copy.deepcopy(decoder_cpu).to(device)
    decoder.eval()
    z_mean = z_mean_cpu.to(device) if torch.is_tensor(z_mean_cpu) else torch.as_tensor(z_mean_cpu, device=device)
    z_std = z_std_cpu.to(device) if torch.is_tensor(z_std_cpu) else torch.as_tensor(z_std_cpu, device=device)
    x_mean_np = _to_numpy_array(X_mean)
    x_std_np = _to_numpy_array(X_std)

    return emulator, decoder, z_mean, z_std, Y_min, Y_max, wave_grid, time_grid, x_mean_np, x_std_np


def clear_typeII_photospheric_model_cache():
    """Clear Photospheric-Model cache to free memory."""
    _load_photospheric_model_cnn_v3.cache_clear()
    _load_photospheric_model_cnn_v3_on_device.cache_clear()



def typeII_spectra_photospheric_model(mass, ni_mass, mixing, energy, Menv, R0, **kwargs):
    """Type II spectra using the Photospheric-Model surrogate."""
    device_key = _canonical_device_key(kwargs.get('device', 'auto'))
    device = torch.device(device_key)
    emulator, decoder, z_mean, z_std, Y_min, Y_max, wave_grid, time_grid, X_mean, X_std = \
        _load_photospheric_model_cnn_v3_on_device(device_key)

    ss = np.array([mass, ni_mass, mixing, energy, Menv, R0]).T
    if isinstance(mass, float):
        ss = ss.reshape(1, -1)
    ss = ((ss - X_mean) / X_std).astype(np.float32)

    with torch.no_grad():
        x_tensor = torch.from_numpy(ss).to(device)
        z_norm = emulator(x_tensor)
        z = z_norm * z_std + z_mean
        pred_norm = decoder(z).squeeze(1).cpu().numpy()

    pred_logf = pred_norm * (Y_max - Y_min) + Y_min
    if isinstance(mass, float):
        pred_logf = pred_logf[0]

    pred_flux = (10 ** pred_logf) * uu.erg / uu.s / uu.Hz
    output = namedtuple('output', ['spectrum', 'frequency', 'time'])
    return output(
        spectrum=pred_flux,
        frequency=np.array(wave_grid) * uu.Angstrom,
        time=np.array(time_grid) * uu.day,
    )
