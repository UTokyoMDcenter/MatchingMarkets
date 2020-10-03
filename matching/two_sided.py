"""
Basic two sided matching markets.

To do: Use a priority queue to hold an entire matching
"""
import numpy as np
import heapq
from .util import InvalidPrefsError, InvalidCapsError


class ManyToOneMarket(object):
    """
    Basic class for the model of a many-to-one two-sided matching market.

    Attributes
    ----------
    num_doctors : int
        The number of doctors.
    
    num_hospitals : int
        The number of hospitals.
    
    doctor_prefs : 2d-array(int)
        The list of doctors' preferences over the hospitals and the outside option.
        The elements must be 0 <= x <= num_hospitals. 
        The number `num_hospitals` is considered as an outside option.

    hospital_prefs : 2d-array(int)
        The list of hospital' preferences over the doctors and the outside option.
        The elements must be 0 <= x <= num_doctors. 
        The number `num_doctors` is considered as an outside option.

    hospital_caps : 1d-array(int, optional)
        The list of the capacities of the hospitals. The elements must be non-negative.
        If nothing is specified, then all caps are set to be 1.
    """
    def __init__(self, doctor_prefs, hospital_prefs, hospital_caps=None):
        self.num_doctors = len(doctor_prefs)
        self.num_hospitals = len(hospital_prefs)
        self.doctor_prefs = doctor_prefs
        self.hospital_prefs = hospital_prefs
        self.doctor_outside_option = self.num_hospitals
        self.hospital_outside_option = self.num_doctors
        self.hospital_caps = hospital_caps
        self._check_prefs()
        self._check_caps()


    def _check_prefs(self):
        """
        Check the validity of preferences.
        """
        try:
            self.doctor_prefs = np.array(self.doctor_prefs, dtype=int)
            self.hospital_prefs = np.array(self.hospital_prefs, dtype=int)
        
        except Exception as e:
            msg = "Each pref must be a matrix of integers.\n" +\
                f"'doctor_prefs': {self.doctor_prefs}\n" +\
                f"'hospital_prefs': {self.hospital_prefs}"
            raise InvalidPrefsError(msg)

        if np.min(self.doctor_prefs) < 0 or \
            np.max(self.doctor_prefs) > self.doctor_outside_option:
            msg = \
                "Elements of 'doctor_prefs' must be 0 <= x <= 'num_hospitals'.\n" +\
                f"'doctor_prefs': {self.doctor_prefs}"
            raise InvalidPrefsError(msg)

        if np.min(self.hospital_prefs) < 0 or \
            np.max(self.hospital_prefs) > self.hospital_outside_option:
            msg = \
                "Elements of 'hospital_prefs' must be 0 <= x <= 'num_doctors'\n" +\
                f"'hospital_prefs': {self.hospital_prefs}"
            raise InvalidPrefsError(msg)


    def _check_caps(self):
        """
        Check the validity of the hospital caps.
        """
        if self.hospital_caps is None:
            self.hospital_caps = np.ones(self.num_hospitals, dtype=int)
        else:
            try:
                self.hospital_caps = np.array(self.hospital_caps, dtype=int)
            
            except Exception as e:
                msg = f"'hospital_caps' must be a list of non-negative integers.\n" +\
                    f"'hospital_caps': {self.hospital_caps}"
                raise InvalidCapsError(msg)

            if len(self.hospital_caps) != self.num_hospitals:
                msg = f"The length of 'hospital_caps' must be equal to 'num_hospitals'.\n" +\
                    f"'hospital_caps': {self.hospital_caps}"
                raise InvalidCapsError(msg)

            if np.any(self.hospital_caps < 0):
                msg = f"'hospital_caps' must be a list of non-negative integers.\n" +\
                    f"'hospital_caps': {self.hospital_caps}"
                raise InvalidCapsError(msg)


    @staticmethod
    def _convert_prefs_to_ranks(prefs, num_objects):
        num_people = len(prefs)
        outside_option = num_objects
        rank_table = np.full([num_people, num_objects], outside_option, dtype=int)

        for p, pref in enumerate(prefs):
            for rank, obj in enumerate(pref):
                if obj == outside_option:
                    break

                rank_table[p, obj] = rank

        return rank_table


    def boston(self, doctor_proposing=True):
        """
        Run Boston algorithm in a many-to-one two-sided matching market.

        By default, this method runs the doctor proposing algorithm 
        and returns a stable matching in the market.

        Args:
            doctor_proposing : bool, optional
                If True, it runs the doctor proposing alg. Otherwise it 
                runs the hospital proposing alg.

        Returns:
            matching : 1d-ndarray
                List of the matched hospitals. The n-th element indicates 
                the hospital which the n-th doctor matches.
        """
        pass


    def deferred_acceptance(self, doctor_proposing=True):
        """
        Run the deferred acceptance (Gale-Shapley) algorithm in 
        a many-to-one two-sided matching market.

        By default, this method runs the doctor proposing DA 
        and returns a stable matching in the market.

        Args:
            doctor_proposing : bool, optional
                If True, it runs the doctor proposing DA. Otherwise it 
                runs the hospital proposing DA.

        Returns:
            matching : 1d-ndarray
                List of the matched hospitals (and the outside option). 
                The n-th element indicates the hospital which 
                the n-th doctor matches.
        """
        if not doctor_proposing:
            raise ValueError("Reverse DA hasn't been implemented yet.")
        
        doctors = list(range(self.num_doctors-1, -1, -1))
        next_proposing_ranks = np.zeros(self.num_doctors, dtype=int)
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)
        matching = np.full(
            self.num_doctors, self.doctor_outside_option, dtype=int)
        worst_matched_doctors = np.full(self.num_hospitals, -1, dtype=int)
        len_d_pref = self.doctor_prefs.shape[1]
        remaining_caps = np.copy(self.hospital_caps)

        while len(doctors) > 0:
            d = doctors.pop()
            first_rank = next_proposing_ranks[d]
            d_pref = self.doctor_prefs[d]

            for rank in range(first_rank, len_d_pref):
                next_proposing_ranks[d] += 1
                h = d_pref[rank]

                # if this doctor's preference list is exhausted
                if h == self.doctor_outside_option:
                    break

                d_rank = hospital_rank_table[h, d]

                # if the doctor's rank is below the outside option
                if d_rank == self.hospital_outside_option:
                    continue

                worst_doctor = worst_matched_doctors[h]

                if worst_doctor == -1:
                    worst_rank = -1
                else:
                    worst_rank = hospital_rank_table[h, worst_doctor]

                # if the cap is not full
                if remaining_caps[h] > 0:
                    matching[d] = h
                    remaining_caps[h] -= 1
                    # update worst rank
                    if d_rank > worst_rank:
                        worst_matched_doctors[h] = d
                    
                    break

                # if the cap is full but a less favorable doctor is matched
                elif d_rank < worst_rank:
                    matching[d] = h
                    matching[worst_doctor] = self.doctor_outside_option
                    doctors.append(worst_doctor)

                    # update worst rank
                    new_worst_doctor = d
                    new_worst_rank = d_rank
                    for dd in np.where(matching == h)[0]:
                        dd_rank = hospital_rank_table[h, dd]
                        if dd_rank > new_worst_rank:
                            new_worst_doctor = dd
                            new_worst_rank = dd_rank

                    worst_matched_doctors[h] = new_worst_doctor
                    break

        return matching


    def check_blocking_pairs(self, matching):
        pass


class OneToOneMarket(ManyToOneMarket):
    """
    Basic class for the model of a one-to-one two-sided matching market.

    Attributes
    ----------
    num_doctors : int
        The number of doctors.
    
    num_hospitals : int
        The number of hospitals.
    
    doctor_prefs : 2d-array(int)
        The list of doctors' preferences over the hospitals and the outside option.
        The elements must be 0 <= x <= num_hospitals. 
        The number `num_hospitals` is considered as an outside option.

    hospital_prefs : 2d-array(int)
        The list of hospital' preferences over the doctors and the outside option.
        The elements must be 0 <= x <= num_doctors. 
        The number `num_doctors` is considered as an outside option.
    """
    def __init__(self, doctor_prefs, hospital_prefs):
        super().__init__(doctor_prefs, hospital_prefs)


if __name__ == "__main__":
    """
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
    m = ManyToOneMarket(d_prefs, h_prefs, caps)
    print(m.deferred_acceptance())
    """

    """
    d_prefs = np.array([
        [2, 0, 4, 3, 5, 1], 
        [0, 2, 3, 1, 4, 5], 
        [3, 4, 2, 0, 1, 5], 
        [2, 3, 0, 4, 5, 1], 
        [0, 3, 1, 5, 2, 4], 
        [3, 2, 1, 0, 4, 5], 
        [1, 4, 0, 2, 5, 3], 
        [0, 2, 1, 4, 3, 5], 
        [3, 0, 4, 5, 1, 2], 
        [2, 0, 4, 1, 3, 5], 
        [4, 3, 0, 2, 1, 5], 
    ])

    h_prefs = np.array([
        [2, 6, 8, 10, 4, 3, 9, 7, 5, 0, 1, 11], 
        [4, 6, 9, 5, 7, 1, 2, 10, 11, 0, 3, 8], 
        [10, 5, 7, 2, 1, 3, 6, 0, 9, 11, 4, 8], 
        [9, 0, 1, 10, 3, 8, 4, 2, 5, 7, 11, 6], 
        [1, 3, 9, 6, 5, 0, 7, 2, 10, 8, 11, 4], 
    ])

    caps = [4, 1, 3, 2, 1]
    m = ManyToOneMarket(d_prefs, h_prefs, caps)
    print(m.deferred_acceptance())
    """

    """
    d_prefs = np.array([
        [2, 0, 4, 3, 5, 1], 
        [0, 2, 3, 1, 4, 5], 
        [3, 4, 2, 0, 1, 5], 
        [2, 3, 0, 4, 5, 1], 
        [0, 3, 1, 5, 2, 4], 
        [3, 2, 1, 0, 4, 5], 
        [1, 4, 0, 2, 5, 3], 
        [0, 2, 1, 4, 3, 5], 
        [3, 0, 4, 5, 1, 2], 
        [2, 0, 4, 1, 3, 5], 
        [4, 3, 0, 2, 1, 5], 
    ])

    h_prefs = np.array([
        [2, 6, 8, 10, 4, 3, 9, 7, 5, 0, 1, 11], 
        [4, 6, 9, 5, 7, 1, 2, 10, 11, 0, 3, 8], 
        [10, 5, 7, 2, 1, 3, 6, 0, 9, 11, 4, 8], 
        [9, 0, 1, 10, 3, 8, 4, 2, 5, 7, 11, 6], 
        [1, 3, 9, 6, 5, 0, 7, 2, 10, 8, 11, 4], 
    ])

    caps = [4, 1, 3, 2, 1]
    regions = [0, 1, 1, 0, 0]
    regional_caps = [3, 2]
    target_caps = [1, 1, 1, 1, 1]
    hospital_order = {
        0: [0, 3, 4], 
        1: [1, 2]
    }
    m = ManyToOneMarketWithRegionalQuotas(d_prefs, h_prefs, caps, regions, regional_caps)
    #print(m.JRMP_mechanism(target_caps))
    print(m.flexible_deferred_acceptance(target_caps, hospital_order))
    """


    # Kamada and Kojima (2010) Example 1 and 6
    """
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
    m = ManyToOneMarketWithRegionalQuotas(d_prefs, h_prefs, caps, regions, regional_caps)
    print("JRMP mechanism result:", m.JRMP_mechanism(target_caps))
    print("flexible DA result:", m.flexible_deferred_acceptance(target_caps, hospital_order))
    """

