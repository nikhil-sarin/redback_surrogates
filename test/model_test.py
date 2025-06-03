import unittest
import numpy as np
from os.path import dirname
from os import listdir
import bilby

import redback_surrogates.model_library

_dirname = dirname(__file__)

class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        self.path_to_files = f"{_dirname}/../redback_surrogates/priors/"
        self.prior_files = listdir(self.path_to_files)

    def tearDown(self) -> None:
        pass

    def get_prior(self, file):
        prior_dict = bilby.prior.PriorDict()
        prior_dict.from_file(f"{self.path_to_files}{file}")
        return prior_dict

    def test_models(self):
        times = np.array([1, 2, 3])
        for f in self.prior_files:
            print(f)
            model_name = f.replace(".prior", "")
            prior = self.get_prior(file=f)
            function = redback_surrogates.model_library.all_models_dict[model_name]
            if model_name in ['typeII']:
                print("Skipping {} model test, as it requires extra files".format(model_name))
                pass
            if model_name == 'tophat_emulator':
                flux_density =  function(times, **prior.sample())
                self.assertEqual(len(times), len(flux_density), flux_density.shape)
            elif model_name in ['bulla_bns_kilonovanet_spectra', 'bulla_nsbh_kilonovanet_spectra', 'kasen_bns_kilonovanet_spectra']:
                spectra = function(times, **prior.sample())
                self.assertEqual((len(spectra.time), len(spectra.lambdas)), np.shape(spectra.spectra))
