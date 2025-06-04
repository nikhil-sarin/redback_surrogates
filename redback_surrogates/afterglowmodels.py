import pickle
from scipy import interpolate
import numpy as np
import os
from redback_surrogates.utils import citation_wrapper
dirname = os.path.dirname(__file__)


with open(f"{dirname}/surrogate_data/tophat_redback_300x3.pkl", "rb") as f:
    model = pickle.load(f)
with open(f"{dirname}/surrogate_data/tophat_redback_scaley.pkl", "rb") as sy:
    scalerY= pickle.load(sy)
with open(f"{dirname}/surrogate_data/tophat_redback_scalex.pkl", "rb") as sx:
    scalerX = pickle.load(sx) 


def _shape_data(thv, loge0, thc, logn0, p, logepse, logepsb, g0,frequency):
    if isinstance(frequency, (int, float)) == True:
        test_data = np.array([np.log10(thv) , loge0 , np.log10(thc), logn0, p, logepse, logepsb, np.log10(g0), frequency]).reshape(1,-1)
    else:
        test_data = []
        for f in frequency:
            test_data.append([np.log10(thv) , loge0 , np.log10(thc), logn0, p, logepse, logepsb, np.log10(g0), f])
    return np.array(test_data)    

    
@citation_wrapper("https://ui.adsabs.harvard.edu/abs/2025MNRAS.539.3319W/abstract")
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
    :param frequency: frequency of the band to view in- single number or same length as time array (in log10 hz)
    :return: flux density at each time for given frequency
    """
    
    frequency = kwargs['frequency']
    test_data = _shape_data(thv, loge0, thc, logn0, p, logepse, logepsb, g0,frequency)
    logtime = np.logspace(2.94,7.41,100)/86400
    
    xtests = scalerX.transform(test_data)
    prediction = model.predict(xtests)
    prediction = np.exp(scalerY.inverse_transform(prediction))
    
    afterglow = interpolate.interp1d(logtime, prediction, kind='linear', fill_value='extrapolate')
    fluxd = afterglow(new_time)
    
    if test_data.shape == (1,9):
        return fluxd[0]
    else:
        return np.diag(fluxd)
