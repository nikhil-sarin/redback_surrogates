# A basic example of how to call the tophat emulator model. Note that ideally you should be calling this model using the redback interface.

import numpy as np
from redback_surrogates import afterglowmodels
import matplotlib.pyplot as plt

# Set up the parameters for the model
tt = np.linspace(0.1, 30, 50)
thv = 0.1
loge0 = 50
thc = 0.1
logn0 = -2
p = 2.2
logepse = -1
logepsb = -2
g0 = 100

# Call the model
out = afterglowmodels.tophat_emulator(tt, thv, loge0, thc, logn0, p, logepse, logepsb, g0, frequency=np.log10(2e14))
plt.loglog(tt, out)
plt.show()
