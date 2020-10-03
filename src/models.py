"""
Models for the two sided matching markets.

To do: Use a priority queue to hold an entire matching
"""
import numpy as np
import heapq


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
        #doctor_outside_option = num_hospitals
        self.hospital_outside_option = self.num_doctors
        #hospital_outside_option = num_doctors
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
        #doctors = list(range(num_doctors-1, -1, -1))
        next_proposing_ranks = np.zeros(self.num_doctors, dtype=int)
        #next_proposing_ranks = np.zeros(num_doctors, dtype=int)
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)
        #hospital_rank_table = _convert_prefs_to_ranks(hospital_prefs, num_doctors)
        matching = np.full(
            self.num_doctors, self.doctor_outside_option, dtype=int)
        #matching = np.full(num_doctors, doctor_outside_option, dtype=int)
        worst_matched_doctors = np.full(self.num_hospitals, -1, dtype=int)
        #worst_matched_doctors = np.full(num_hospitals, -1, dtype=int)
        len_d_pref = self.doctor_prefs.shape[1]
        #len_d_pref = doctor_prefs.shape[1]
        remaining_caps = np.copy(self.hospital_caps)
        #remaining_caps = np.copy(hospital_caps)

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


class ManyToOneMarketWithRegionalQuotas(ManyToOneMarket):
    """
    Class for the model of a many-to-one two-sided matching market
    with regional quotas.

    Attributes
    ----------
    num_doctors : int
        The number of doctors.

    num_hospitals : int
        The number of hospitals.

    num_regions : int
        The number of regions.

    doctor_prefs : 2d-array(int)
        The list of doctors' preferences over the hospitals and the outside option.
        The elements must be 0 <= x <= num_hospitals.
        The number `num_hospitals` is considered as an outside option.

    hospital_prefs : 2d-array(int)
        The list of hospital' preferences over the doctors and the outside option.
        The elements must be 0 <= x <= num_doctors.
        The number `num_doctors` is considered as an outside option.

    hospital_caps : 1d-array(int)
        The list of the capacities of the hospitals. The elements must be non-negative.

    hospital_regions : 1d-array(int)
        The list of regions each hospital belongs to.

    regional_caps : 1d-array(int)
        The list of the capacities of each region. The elements must be non-negative.
    """
    def __init__(self,
        doctor_prefs,
        hospital_prefs,
        hospital_caps,
        hospital_regions,
        regional_caps
        ):
        super().__init__(doctor_prefs, hospital_prefs, hospital_caps)
        self.num_regions = len(np.unique(hospital_regions))
        self.hospital_regions = hospital_regions
        self.regional_caps = regional_caps
        self._check_regions()


    def _check_regions(self):
        """
        Check the validity of the hospital regions and regional caps.
        """
        # hospital regions
        try:
            self.hospital_regions = np.array(self.hospital_regions, dtype=int)

        except Exception as e:
            msg = f"'hospital_regions' must be a list of integers.\n" +\
                f"'hospital_regions': {self.hospital_regions}"
            raise InvalidRegionError(msg)

        if len(self.hospital_regions) != self.num_hospitals:
            msg = f"The length of 'hospital_regions' must be equal to 'num_hospitals'.\n" +\
                f"'hospital_regions': {self.hospital_regions}"
            raise InvalidRegionError(msg)

        # regional caps
        try:
            self.regional_caps = np.array(self.regional_caps, dtype=int)

        except Exception as e:
            msg = f"'regional_caps' must be a list of non-negative integers.\n" +\
                f"'regional_caps': {self.regional_caps}"
            raise InvalidCapsError(msg)

        if len(self.regional_caps) != self.num_regions:
            msg = f"The length of 'regional_caps' must be equal to 'num_regions'.\n" +\
                f"'regional_caps': {self.regional_caps}"
            raise InvalidCapsError(msg)

        if np.any(self.regional_caps < 0):
            msg = f"'regional_caps' must be a list of non-negative integers.\n" +\
                f"'regional_caps': {self.regional_caps}"
            raise InvalidCapsError(msg)


    def _check_target_caps(self, target_caps):
        """
        Check the validity of the regional caps.
        """
        try:
            target_caps = np.array(target_caps, dtype=int)

        except Exception as e:
            msg = f"'target_caps' must be a list of non-negative integers.\n" +\
                f"'target_caps': {target_caps}"
            raise InvalidCapsError(msg)

        if len(target_caps) != self.num_hospitals:
            msg = f"The length of 'target_caps' must be equal to 'num_hospitals'.\n" +\
                f"'target_caps': {target_caps}"
            raise InvalidCapsError(msg)

        if np.any(self.regional_caps < 0):
            msg = f"'target_caps' must be a list of non-negative integers.\n" +\
                f"'target_caps': {target_caps}"
            raise InvalidCapsError(msg)

        # check whether the sum of target caps in a region is less than
        # or equal to its regional cap
        remaining_regional_caps = np.copy(self.regional_caps)
        for h in range(self.num_hospitals):
            h_region = self.hospital_regions[h]
            remaining_regional_caps[h_region] -= target_caps[h]

        if np.any(remaining_regional_caps < 0):
            msg = "The sum of the target capacities of the hospitals in each " + \
                "region must be less than or equal to its regional quota."
            raise InvalidCapsError(msg)

        return target_caps


    def _check_hospital_order(self, hospital_order):
        """
        Implemented later
        """
        return hospital_order


    def JRMP_mechanism(self, target_caps):
        """
        Run the JRMP mechanism introduced in Kamada and Kojima (2010)
        in the market with regional quotas.

        Args:
            target_caps : 1d-array(int)
                List of the target capacities of the hospitals.
                The sum of the target capacities of the hospitals in each
                region must be less than or equal to its regional quota.

        Returns:
            matching : 1d-array(int)
                List of the matched hospitals (and the outside option).
                The n-th element indicates the hospital which
                the n-th doctor matches.
        """
        target_caps = self._check_target_caps(target_caps)
        original_caps = self.hospital_caps
        self.hospital_caps = target_caps
        matching = self.deferred_acceptance()
        self.hospital_caps = original_caps
        return matching


    def flexible_deferred_acceptance(self, target_caps, hospital_order):
        """
        Run the flexible deferred acceptance algorithm proposed in
        Kamada and Kojima (2010) in the market with regional quotas.

        Args:
            target_caps : 1d-array(int)
                List of the target capacities of the hospitals.
                The sum of the target capacities of the hospitals in each
                region must be less than or equal to its regional quota.

            hospital_order : dict
                Order of hospitals in each region.
                {region: [order of hospitals in the region]}

        Returns:
            matching : 1d-array(int)
                List of the matched hospitals (and the outside option).
                The n-th element indicates the hospital which
                the n-th doctor matches.
        """
        target_caps = self._check_target_caps(target_caps)
        hospital_order = self._check_hospital_order(hospital_order)

        doctors = list(range(self.num_doctors-1, -1, -1))
        next_proposing_ranks = np.zeros(self.num_doctors, dtype=int)
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)

        #print(hospital_rank_table)

        matching = np.full(
            self.num_doctors, self.doctor_outside_option, dtype=int)
        worst_doctors_in_target_caps = np.full(self.num_hospitals, -1, dtype=int)
        len_d_pref = self.doctor_prefs.shape[1]
        remaining_target_caps = np.copy(target_caps)
        remaining_regional_caps = np.copy(self.regional_caps)
        adjustment_matching = {h: [] for h in range(self.num_hospitals)}

        while len(doctors) > 0:
            d = doctors.pop()
            d_pref = self.doctor_prefs[d]

            for rank in range(next_proposing_ranks[d], len_d_pref):
                next_proposing_ranks[d] += 1
                h = d_pref[rank]

                # if this doctor's preference list is exhausted
                if h == self.doctor_outside_option:
                    break

                d_rank = hospital_rank_table[h, d]
                h_region = self.hospital_regions[h]
                #print("d:", d, "h:", h,"d_rank:", d_rank, "h_region:", h_region, "doctors:", doctors)

                # if the doctor's rank is below the outside option
                if d_rank == self.hospital_outside_option:
                    continue

                else:
                    worst_doctor = worst_doctors_in_target_caps[h]

                    if worst_doctor == -1:
                        worst_rank = -1
                    else:
                        worst_rank = hospital_rank_table[h, worst_doctor]

                    # if the target cap is not full, it accepts
                    if remaining_target_caps[h] > 0:
                        matching[d] = h
                        remaining_target_caps[h] -= 1
                        remaining_regional_caps[h_region] -= 1
                        # update worst rank
                        if d_rank > worst_rank:
                            worst_doctors_in_target_caps[h] = d

                        #print("matching:", matching)
                        break

                    # if the target cap is full but a less favorable doctor is matched
                    # in the target cap, then it accepts a new doctor and the worst
                    # doctor goes to the adjustment matching step
                    elif d_rank < worst_rank:
                        matching[d] = h
                        matching[worst_doctor] = self.doctor_outside_option
                        heapq.heappush(adjustment_matching[h], worst_rank)

                        # update worst rank
                        new_worst_doctor = d
                        new_worst_rank = d_rank
                        for dd in np.where(matching == h)[0]:
                            dd_rank = hospital_rank_table[h, dd]
                            if dd_rank > new_worst_rank:
                                new_worst_doctor = dd
                                new_worst_rank = dd_rank

                        worst_doctors_in_target_caps[h] = new_worst_doctor

                    else:
                        heapq.heappush(adjustment_matching[h], d_rank)

                    #print("matching:", matching)

                    # adjustment matching step
                    #print("adj bf:", adjustment_matching)
                    hopitals_in_same_region = hospital_order[h_region]
                    hospitals = hopitals_in_same_region[:]
                    new_adjustment_matching = {hh: [] for hh in hopitals_in_same_region}
                    adjustment_caps = self.hospital_caps - target_caps
                    remaining_regional_cap = remaining_regional_caps[h_region]
                    num_matches = 0
                    while len(hospitals) > 0:
                        hh = hospitals.pop(0)

                        if num_matches >= remaining_regional_cap:
                            break

                        if len(adjustment_matching[hh]) == 0:
                            continue

                        if adjustment_caps[hh] == 0:
                            continue

                        dd_rank = heapq.heappop(adjustment_matching[hh])
                        heapq.heappush(new_adjustment_matching[hh], dd_rank)
                        adjustment_caps[hh] -= 1
                        num_matches += 1

                        hospitals.append(hh)

                    # an unmatched doctor moves to the original queue
                    for hh in hopitals_in_same_region:
                        if len(adjustment_matching[hh]) > 0:
                            dd_rank = heapq.heappop(adjustment_matching[hh])
                            doctors.append(self.hospital_prefs[hh, dd_rank])

                    # substitute new heaps to the original dict
                    for hh in hopitals_in_same_region:
                        adjustment_matching[hh] = new_adjustment_matching[hh]

                    #print("adj af:", adjustment_matching)
                    break

        # substitute the adjustment matching to the original
        for h in range(self.num_hospitals):
            for _ in range(len(adjustment_matching[h])):
                d_rank = heapq.heappop(adjustment_matching[h])
                d = self.hospital_prefs[h, d_rank]
                matching[d] = h

        return matching


if __name__ == "__main__":
    """
    doctor_prefs = [
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [2, 0, 1],
    ]
    hospital_prefs = [
        [0, 2, 1, 3],
        [1, 0, 2, 3],
        [2, 0, 3, 1],
    ]
    hospital_caps = np.array([1, 1, 1])
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
