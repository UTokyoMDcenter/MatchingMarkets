"""
Utility classes and functions used in this library

"""

class InvalidPrefsError(Exception):
    """
    Exception called when input preferences are invalid.
    """
    pass


class InvalidCapsError(Exception):
    """
    Exception called when input caps are invalid.
    """
    pass


class InvalidRegionError(Exception):
    """
    Exception called when input regions are invalid.
    """
    pass


def generate_prefs(
    num_doctors, 
    num_hospitals, 
    outside_option=False, 
    random_seed=None
    ):
    """
    Randomly generate preference lists of doctors and hospitals.

    """
    random_state = np.random.RandomState(seed=random_seed)

    if outside_option:
        len_d_pref = num_hospitals
        len_h_pref = num_doctors
    else:
        len_d_pref = num_hospitals + 1
        len_h_pref = num_doctors + 1

    # some 

    return d_prefs, h_prefs
    