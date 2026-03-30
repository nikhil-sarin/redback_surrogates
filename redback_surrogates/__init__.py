from redback_surrogates import utils, afterglowmodels, data_management
import sys

# Lazy import to avoid loading keras/tensorflow/kilonovanet unless needed
_LAZY_MODULES = {
    "supernovamodels": "redback_surrogates.supernovamodels",
    "model_library": "redback_surrogates.model_library",
    "kilonovamodels": "redback_surrogates.kilonovamodels",
}

def __getattr__(name):
    if name in _LAZY_MODULES:
        import importlib
        module = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")