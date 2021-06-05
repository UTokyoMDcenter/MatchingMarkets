"""
Two sided matching markets with regional quotas.

"""
import numpy as np
import heapq
import copy
from MatchingMarkets.util import InvalidPrefsError, InvalidCapsError, MaxHeap, MinHeap, \
    generate_random_prefs, generate_caps_given_sum, round_caps_to_meet_sum
from MatchingMarkets.two_sided import ManyToOneMarket


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
        self.hospital_regions = hospital_regions
        self.regional_caps = regional_caps
        self._check_regions()


    def _check_regions(self):
        """
        Check the validity of the hospital regions and regional caps.
        """
        # regional caps
        try:
            self.regional_caps = np.array(self.regional_caps, dtype=int)
            self.num_regions = len(self.regional_caps)
        
        except Exception as e:
            msg = f"'regional_caps' must be a list of non-negative integers.\n" +\
                f"'regional_caps': {self.regional_caps}"
            raise InvalidCapsError(msg, "regional_caps")

        if np.any(self.regional_caps < 0):
            msg = f"'regional_caps' must be a list of non-negative integers.\n" +\
                f"'regional_caps': {self.regional_caps}"
            raise InvalidCapsError(msg, "regional_caps")

        # hospital regions
        try:
            self.hospital_regions = np.array(self.hospital_regions, dtype=int)
        
        except Exception as e:
            msg = f"'hospital_regions' must be a list of integers.\n" +\
                f"'hospital_regions': {self.hospital_regions}"
            raise InvalidRegionError(msg, "hospital_regions")

        if len(self.hospital_regions) != self.num_hospitals:
            msg = f"The length of 'hospital_regions' must be equal to 'num_hospitals'.\n" +\
                f"'hospital_regions': {self.hospital_regions}"
            raise InvalidRegionError(msg, "hospital_regions")

        if np.min(self.hospital_regions) < 0 or np.max(self.hospital_regions) >= self.num_regions:
            msg = f"The elements of 'hospital_regions' must 0 <= x < 'num_regions'.\n" +\
                f"'hospital_regions': {self.hospital_regions}"
            raise InvalidRegionError(msg, "hospital_regions")


    def _check_target_caps(self, target_caps):
        """
        Check the validity of the regional caps.
        """
        try:
            target_caps = np.array(target_caps, dtype=int)
        
        except Exception as e:
            msg = f"'target_caps' must be a list of non-negative integers.\n" +\
                f"'target_caps': {target_caps}"
            raise InvalidCapsError(msg, "target_caps")

        if len(target_caps) != self.num_hospitals:
            msg = f"The length of 'target_caps' must be equal to 'num_hospitals'.\n" +\
                f"'target_caps': {target_caps}"
            raise InvalidCapsError(msg, "target_caps")

        if np.any(self.regional_caps < 0):
            msg = f"'target_caps' must be a list of non-negative integers.\n" +\
                f"'target_caps': {target_caps}"
            raise InvalidCapsError(msg, "target_caps")

        # check whether the sum of target caps in a region is less than
        # or equal to its regional cap
        remaining_regional_caps = np.copy(self.regional_caps)
        for h in range(self.num_hospitals):
            h_region = self.hospital_regions[h]
            remaining_regional_caps[h_region] -= target_caps[h]

        if np.any(remaining_regional_caps < 0):
            msg = "The sum of the target capacities of the hospitals in each " + \
                "region must be less than or equal to its regional quota."
            raise InvalidCapsError(msg, "target_caps")

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
        
        # for in target caps matching process
        matched_doctor_ranks_in_target_caps = \
            {h: MaxHeap(self.num_doctors) for h in range(self.num_hospitals)}
        
        # for adjustment process
        matched_doctor_ranks_in_adjustment = \
            {h: MinHeap(self.num_doctors) for h in range(self.num_hospitals)}

        # count consumed caps in target caps matching process
        remaining_regional_caps = np.copy(self.regional_caps)

        while len(doctors) > 0:
            d = doctors.pop()
            first_rank = next_proposing_ranks[d]
            d_pref = self.doctor_prefs[d]

            for rank in range(first_rank, self.num_hospitals+1):
                next_proposing_ranks[d] += 1
                h = d_pref[rank]

                # if this doctor's preference list is exhausted
                if h == self.doctor_outside_option:
                    break

                d_rank = hospital_rank_table[h, d]
                h_region = self.hospital_regions[h]

                # if the doctor's rank is below the outside option
                if d_rank == self.hospital_outside_option:
                    continue

                """
                Need to retry adjustment matching process after here
                """
                # if the original cap is 0
                if self.hospital_caps[h] == 0:
                    continue

                # if the target cap is 0
                if target_caps[h] == 0:
                    matched_doctor_ranks_in_adjustment[h].push(d_rank)

                # if the target cap is not full, it accepts
                elif matched_doctor_ranks_in_target_caps[h].length < target_caps[h]:
                    matched_doctor_ranks_in_target_caps[h].push(d_rank)
                    remaining_regional_caps[h_region] -= 1

                # if the target cap is full but a less favorable doctor is matched  
                # in the target cap, then it accepts a new doctor and the worst 
                # doctor goes to the adjustment matching step
                elif d_rank < matched_doctor_ranks_in_target_caps[h].root():
                    worst_rank = matched_doctor_ranks_in_target_caps[h].replace(d_rank)
                    matched_doctor_ranks_in_adjustment[h].push(worst_rank)

                else:
                    matched_doctor_ranks_in_adjustment[h].push(d_rank)

                # adjustment matching step
                hopitals_in_same_region = copy.deepcopy(hospital_order[h_region])
                new_matched_doctor_ranks_in_adjustment = \
                    {hh: MinHeap(self.num_doctors) for hh in hopitals_in_same_region}
                remaining_regional_cap = remaining_regional_caps[h_region]
                hospital_caps_in_adjustment = self.hospital_caps - target_caps
                num_matches = 0
                
                while len(hopitals_in_same_region) > 0:
                    hh = hopitals_in_same_region.pop(0)
                    
                    # if remaining regional cap is full
                    if num_matches >= remaining_regional_cap:
                        break

                    # if no doctor is proposing to hh
                    if matched_doctor_ranks_in_adjustment[hh].length == 0:
                        continue

                    # if hospital cap is not remaining
                    if hospital_caps_in_adjustment[hh] == 0:
                        continue

                    dd_rank = matched_doctor_ranks_in_adjustment[hh].pop()
                    new_matched_doctor_ranks_in_adjustment[hh].push(dd_rank)
                    hospital_caps_in_adjustment[hh] -= 1
                    num_matches += 1
                    hopitals_in_same_region.append(hh)

                # an unmatched doctor moves to the original queue
                for hh in hospital_order[h_region]:
                    if matched_doctor_ranks_in_adjustment[hh].length > 0:
                        # for debug
                        if matched_doctor_ranks_in_adjustment[hh].length > 1:
                            raise ValueError("Unexpected adjustment process occurs")
                        
                        dd_rank = matched_doctor_ranks_in_adjustment[hh].pop()
                        doctors.append(self.hospital_prefs[hh, dd_rank])

                # substitute new heaps to the original dict
                for hh in hospital_order[h_region]:
                    matched_doctor_ranks_in_adjustment[hh] = \
                        new_matched_doctor_ranks_in_adjustment[hh]

                break

        matching = self._convert_matching_heap_to_list(
            matched_doctor_ranks_in_target_caps)

        # substitute the adjustment matching to the original
        for h in range(self.num_hospitals):
            for _ in range(matched_doctor_ranks_in_adjustment[h].length):
                d_rank = matched_doctor_ranks_in_adjustment[h].pop()
                d = self.hospital_prefs[h, d_rank]
                matching[d] = h

        return matching


    @staticmethod
    def create_target_caps(hospital_caps, hospital_regions, regional_caps):
        hospital_caps = np.copy(hospital_caps)
        regions = np.unique(hospital_regions)
        for r in regions:
            hospitals = np.where(hospital_regions == r)[0]
            modified_caps = round_caps_to_meet_sum(
                hospital_caps[hospitals], regional_caps[r])
            hospital_caps[hospitals] = modified_caps

        return hospital_caps


    @staticmethod
    def create_hospital_order(num_regions, hospital_regions):
        hospital_order = {i: [] for i in range(num_regions)}
        for h, r in enumerate(hospital_regions):
            hospital_order[r].append(h)

        return hospital_order


    @staticmethod
    def create_setup(
        num_doctors, 
        num_hospitals, 
        num_regions, 
        outside_option=False, 
        random_seed=None
        ):
        random_generator = np.random.default_rng(seed=random_seed)
        setup = {}
        
        setup["d_prefs"] = generate_random_prefs(
            num_doctors, 
            num_hospitals, 
            outside_option,
            random_generator
        )

        setup["h_prefs"] = generate_random_prefs(
            num_hospitals, 
            num_doctors, 
            outside_option,
            random_generator
        )
        
        setup["hospital_caps"] = generate_caps_given_sum(
            num_hospitals, int(num_doctors*3/2), random_generator)
        
        setup["hospital_regions"] = random_generator.integers(
            0, num_regions, size=num_hospitals)
        
        setup["regional_caps"] = generate_caps_given_sum(
            num_regions, num_doctors, random_generator)

        setup["target_caps"] = ManyToOneMarketWithRegionalQuotas.create_target_caps(
            setup["hospital_caps"], 
            setup["hospital_regions"], 
            setup["regional_caps"]
        )

        setup["hospital_order"] = ManyToOneMarketWithRegionalQuotas.create_hospital_order(
            num_regions, setup["hospital_regions"])

        return setup


    def compute_regional_counts(self, matching):
        regional_counts = [0 for r in range(self.num_regions)]
        for d, h in enumerate(matching):
            if h != self.doctor_outside_option:
                h_region = self.hospital_regions[h]
                regional_counts[h_region] += 1

        return regional_counts


    def check_regional_caps(self, matching):
        regional_counts = self.compute_regional_counts(matching)
        for r, c in enumerate(regional_counts):
            if self.regional_caps[r] < c:
                return False

        return True


    def get_illegitimate_blocking_pair_doctors(
        self, 
        matching, 
        target_caps=None, 
        stability_notion="weak"
        ):
        """
        Count blocking pairs which is illegitimate in the definition 
        of each stability definition in Kamada and Kojima (2015)
        """
        if stability_notion not in ["weak", "standard"]:
            raise ValueError(
                f"Blocking pair count under stability_notion: "+\
                f"'{stability_notion}' is not implemented.")

        if stability_notion == "standard":
            if target_caps is None:
                raise ValueError(
                    f"'target_caps' should be set when stability_notion == 'standard'"
                )

            target_caps = self._check_target_caps(target_caps)

        blocking_pairs = []
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)

        # compute regional counts
        regional_counts = self.compute_regional_counts(matching)

        # compute hospital matching
        matched_doctor_ranks = {
            h: MaxHeap(self.hospital_caps[h]) for h in range(self.num_hospitals)
        }
        for d, h in enumerate(matching):
            if h != self.doctor_outside_option:
                matched_doctor_ranks[h].push(hospital_rank_table[h][d])

        for d, h in enumerate(matching):
            for new_h in self.doctor_prefs[d]:
                # if blocking pair including d does not exist
                if new_h == h:
                    break
                
                # if new_h is unacceptable for d
                if new_h == self.doctor_outside_option:
                    blocking_pairs.append([d, new_h])
                    break

                d_rank = hospital_rank_table[new_h, d]

                # if d is unacceptable for new_h
                if d_rank == self.hospital_outside_option:
                    continue

                # if cap of new_h is 0, then (d, new_h) is not a blocking pair
                if self.hospital_caps[new_h] == 0:
                    continue

                new_h_region = self.hospital_regions[new_h]
                is_blocking_pair = False
                
                # if cap of new_h is not full
                if matched_doctor_ranks[new_h].length < self.hospital_caps[new_h]:
                    is_blocking_pair = True

                # if worse rank doctor is matched
                elif d_rank < matched_doctor_ranks[new_h].root():
                    is_blocking_pair = True

                if is_blocking_pair:
                    if matched_doctor_ranks[new_h].length == 0:
                        second_cond = True
                    else:
                        second_cond = (d_rank > matched_doctor_ranks[new_h].root())

                    # weak stability case
                    if stability_notion == "weak":
                        first_cond = (
                            regional_counts[new_h_region] == \
                            self.regional_caps[new_h_region]
                        )
                        if not (first_cond and second_cond):
                            blocking_pairs.append([d, new_h])
                            break

                    elif stability_notion == "standard":
                        first_cond = (
                            regional_counts[new_h_region] == \
                            self.regional_caps[new_h_region]
                        )

                        if h == self.doctor_outside_option:
                            third_cond = True

                        else:
                            h_region = self.hospital_regions[h]
                            third_cond_1 = (new_h_region != h_region)
                            third_cond_2 = \
                                (matched_doctor_ranks[new_h].length + 1 - target_caps[new_h]) \
                                > (matched_doctor_ranks[h].length - 1 - target_caps[h])

                            third_cond = third_cond_1 or third_cond_2

                        if not (first_cond and second_cond and third_cond):
                            blocking_pairs.append([d, new_h])
                            break

            # if new_h is unacceptable for d
            else:
                blocking_pairs.append([d, -1])

        return blocking_pairs


    def get_num_regional_cap_violations(self, matching):
        regional_counts = np.zeros_like(self.regional_caps)
        for h in matching:
            if h != self.doctor_outside_option:
                regional_counts[self.hospital_regions[h]] += 1

        num_violations = regional_counts - self.regional_caps
        num_violations[num_violations < 0] = 0

        return num_violations


    def analyze_matching(self, matching, target_caps=None):
        result = {}

        hospital_matching = {h: [] for h in range(self.num_hospitals+1)}
        for d, h in enumerate(matching):
            hospital_matching[h].append(d)

        result["unmatch_doctor_size"] = len(hospital_matching[self.doctor_outside_option])
        result["matching_size"] = self.num_doctors - result["unmatch_doctor_size"]
        result["unmatch_hospital_cap_size"] = \
            np.sum(self.hospital_caps) - result["matching_size"]
        result["cap_full_hospital_size"] = 0
        for h in range(self.num_hospitals):
            if len(hospital_matching[h]) == self.hospital_caps[h]:
                result["cap_full_hospital_size"] += 1

        blocking_pair_doctors = self.get_illegitimate_blocking_pair_doctors(
            matching, 
            target_caps=target_caps, 
            stability_notion="standard"
        )

        result["blocking_pair_size"] = len(blocking_pair_doctors)
        result["num_violations"] = np.sum(
            self.get_num_regional_cap_violations(matching))

        matching_ranks = self.get_doctor_matching_ranks(matching)
        result["num_rank1_doctors"] = len(matching_ranks[matching_ranks == 0])
        result["num_rank12_doctors"] = len(matching_ranks[matching_ranks <= 1])

        return result


if __name__ == "__main__":
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
    JRMP_matching = m.JRMP_mechanism(target_caps)
    FDA_matching = m.flexible_deferred_acceptance(target_caps, hospital_order)
    print(JRMP_matching, m.analyze_matching(JRMP_matching))
    print(FDA_matching, m.analyze_matching(FDA_matching))
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
    JRMP_matching = m.JRMP_mechanism(target_caps)
    FDA_matching = m.flexible_deferred_acceptance(target_caps, hospital_order)
    print(JRMP_matching, m.analyze_matching(JRMP_matching))
    print(FDA_matching, m.analyze_matching(FDA_matching))
    """
