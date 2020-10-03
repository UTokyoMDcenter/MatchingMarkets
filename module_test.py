"""
For import test (temporary)

"""

import numpy as np
import matching


if __name__ == "__main__":
    d_prefs = [
        [0, 2, 1], 
        [1, 0, 2], 
        [0, 1, 2], 
        [2, 0, 1], 
    ]
    h_prefs = [
        [0, 2, 1, 3], 
        [1, 0, 2, 3], 
        [2, 0, 3, 1], 
    ]
    caps = np.array([1, 1, 1])
    m = matching.ManyToOneMarket(d_prefs, h_prefs, caps)
    print("DA result:", m.deferred_acceptance())


    num_doctors = 10
    num_hospitals = 2

    d_prefs = np.array([
        [0, 2, 1] for i in range(3) 
    ] + [
        [1, 2, 0] for i in range(num_doctors-3) 
    ])

    h_prefs = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        for i in range(num_hospitals)
    ])

    print("d_prefs:", d_prefs)
    print("h_prefs:", h_prefs)

    caps = [10, 10]
    regions = [0, 0]
    regional_caps = [10]
    target_caps = [5, 5]
    hospital_order = {
        0: [0, 1]
    }
    m = matching.ManyToOneMarketWithRegionalQuotas(
        d_prefs, h_prefs, caps, regions, regional_caps)
    print("JRMP mechanism result:", m.JRMP_mechanism(target_caps))
    print("flexible DA result:", m.flexible_deferred_acceptance(target_caps, hospital_order))


