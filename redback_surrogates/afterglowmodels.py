import pickle
from scipy import interpolate
import numpy as np
import os
from redback_surrogates.utils import citation_wrapper
dirname = os.path.dirname(__file__)

with open(f"{dirname}/surrogate_data/onax_redback.pkl", "rb") as f_on:
    model_on = pickle.load(f_on)
with open(f"{dirname}/surrogate_data/onax_scalery.pkl", "rb") as sy_on:
    scalery_on = pickle.load(sy_on)
with open(f"{dirname}/surrogate_data/onax_scalerx.pkl", "rb") as sx_on:
    scalerx_on = pickle.load(sx_on) 
    
with open(f"{dirname}/surrogate_data/offax_redback.pkl", "rb") as f_off:
    model_off = pickle.load(f_off)
with open(f"{dirname}/surrogate_data/offax_scalery.pkl", "rb") as sy_off:
    scalery_off = pickle.load(sy_off)
with open(f"{dirname}/surrogate_data/offax_scalerx.pkl", "rb") as sx_off:
    scalerx_off = pickle.load(sx_off) 

def _shape_data(thv, loge0, thc, logn0, p, logepse, logepsb, g0,frequency):
    if isinstance(frequency, (int, float)) == True:
        test_data = np.array([np.log10(thv) , loge0 , np.log10(thc), logn0, p, logepse, logepsb, np.log10(g0), frequency]).reshape(1,-1)
    else:
        test_data = []
        for f in frequency:
            test_data.append([np.log10(thv) , loge0 , np.log10(thc), logn0, p, logepse, logepsb, np.log10(g0), f])
    return np.array(test_data)    

@citation_wrapper("Wallace and Sarin in prep.")
def tophat_emulator(new_time, thv, loge0, thc, logn0, p, logepse, logepsb, g0, **kwargs):
    """
    tophat afterglow model using trained mpl regressor

    :param new_time: time in days in observer frame to evaluate at
    :param thv: viewing angle in radians
    :param loge0: log10 on axis isotropic equivalent energy
    :param thc: half width of jet core/jet opening angle in radians
    :param logn0: log10 number density of ISM in cm^-3
    :param p: electron distribution power law index. Must be greater than 2.
    :param logepse: log10 fraction of thermal energy in electrons
    :param logepsb: log10 fraction of thermal energy in magnetic field
    :param g0: initial lorentz factor
    :param kwargs: extra arguments for the model
    :param frequency: frequency of the band to view in- single number or same length as time array
    :return: flux density at each time for given frequency
    """

    frequency = kwargs['frequency']
    test_data = _shape_data(thv, loge0, thc, logn0, p, logepse, logepsb, g0,frequency)
    logtime = np.logspace(2.94,7.41,100)/86400

    if thv <= thc:
        xtests = scalerx_on.transform(test_data)
        prediction = model_on.predict(xtests)
        prediction = np.exp(scalery_on.inverse_transform(prediction))
    else:
        xtests = scalerx_off.transform(test_data)
        prediction = model_off.predict(xtests)
        prediction = np.exp(scalery_off.inverse_transform(prediction))

    afterglow = interpolate.interp1d(logtime, prediction, kind='linear', fill_value='extrapolate')
    fluxd = afterglow(new_time)
    
    if test_data.shape == (1,9):
        return fluxd[0]
    else:
        return np.diag(fluxd)
