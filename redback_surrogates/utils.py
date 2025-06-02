def citation_wrapper(r):
    """
    Wrapper for citation function to allow functions to have a citation attribute
    :param r: proxy argument
    :return: wrapped function
    """
    def wrapper(f):
        f.citation = r
        return f
    return wrapper

def convert_to_observer_frame(time_source_frame, wavelength_source_frame, redshift):
    """
    Convert time and wavelength from source frame to observer frame

    :param time_source_frame: time in source frame
    :param wavelength_source_frame: wavelength in source frame
    :param redshift: redshift of source
    :return: time in observer frame, wavelength in observer frame
    """
    time = time_source_frame * (1 + redshift)
    wavelength = wavelength_source_frame * (1 + redshift)
    return time, wavelength

def get_priors(model):
    from bilby.core.prior import PriorDict
    import os
    import logging

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('redback_surrogates.utils')

    priors = PriorDict()
    try:
        filename = os.path.join(os.path.dirname(__file__), 'priors', f'{model}.prior')
        priors.from_file(filename)
    except FileNotFoundError:
        logger.warning(f'No prior file found for model {model}. '
                    f'Perhaps you also want to set up the prior for the base model? '
                    f'Or you may need to set up your prior explicitly.')
        logger.info('Returning Empty PriorDict.')
    return priors