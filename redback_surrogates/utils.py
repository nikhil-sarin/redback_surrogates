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