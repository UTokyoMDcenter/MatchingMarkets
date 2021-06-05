"""
Basic two sided matching markets.

"""
import numpy as np
from MatchingMarkets.util import InvalidPrefsError, InvalidCapsError, MaxHeap, \
    generate_prefs_from_random_scores, generate_caps_given_sum, round_caps_to_meet_sum
from MatchingMarkets.matching_alg import deferred_acceptance


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
        The list of doctors' preference list over the hospitals and the outside option.
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
    def __init__(self, doctor_prefs, hospital_prefs, hospital_caps=None, no_validation=False):
        self.num_doctors = len(doctor_prefs)
        self.num_hospitals = len(hospital_prefs)
        self.doctor_outside_option = self.num_hospitals
        self.hospital_outside_option = self.num_doctors
        self.doctor_prefs = doctor_prefs
        self.hospital_prefs = hospital_prefs

        if hospital_caps is None:
            self.hospital_caps = np.ones(self.num_hospitals, dtype=int)
        else:
            self.hospital_caps = hospital_caps

        if not no_validation:
            self._convert_prefs()
            self._convert_caps()


    def _convert_prefs(self):
        """
        Check the validity of doctor_prefs and hospital_prefs and
        convert them into 2d-ndarrays.
        """
        converted_doctor_prefs = np.full(
            (self.num_doctors, self.num_hospitals+1), 
            self.doctor_outside_option, 
            dtype=int
        )
        converted_hospital_prefs = np.full(
            (self.num_hospitals, self.num_doctors+1), 
            self.hospital_outside_option, 
            dtype=int
        )

        # doctor prefs validation
        try:
            for d, d_pref in enumerate(self.doctor_prefs):
                for rank, h in enumerate(d_pref):
                    converted_doctor_prefs[d, rank] = h

        except IndexError as e:
            msg = "A preference list is too long.\n" + \
                f"'doctor_prefs': {self.doctore_prefs}"
            raise InvalidPrefsError(msg, "doctor_prefs")
            
        except Exception as e:
            msg = "Each pref must be a matrix of integers.\n" + \
                f"'doctor_prefs': {self.doctor_prefs}"
            raise InvalidPrefsError(msg, "doctor_prefs")

        if np.min(converted_doctor_prefs) < 0 or \
            np.max(converted_doctor_prefs) > self.doctor_outside_option:
            msg = \
                "Elements of 'doctor_prefs' must be 0 <= x <= 'num_hospitals'.\n" +\
                f"'doctor_prefs': {self.doctor_prefs}"
            raise InvalidPrefsError(msg, "doctor_prefs")

        # hospital prefs validation
        try:
            for h, h_pref in enumerate(self.hospital_prefs):
                for rank, d in enumerate(h_pref):
                    converted_hospital_prefs[h, rank] = d

        except IndexError as e:
            msg = "A preference list is too long.\n" +\
                f"'hospital_prefs': {self.hospital_prefs}"
            raise InvalidPrefsError(msg, "hospital_prefs")
            
        except Exception as e:
            msg = "Each pref must be a matrix of integers.\n" +\
                f"'hospital_prefs': {self.hospital_prefs}"
            raise InvalidPrefsError(msg, "hospital_prefs")

        if np.min(converted_hospital_prefs) < 0 or \
            np.max(converted_hospital_prefs) > self.hospital_outside_option:
            msg = \
                "Elements of 'hospital_prefs' must be 0 <= x <= 'num_doctors'\n" +\
                f"'hospital_prefs': {self.hospital_prefs}"
            raise InvalidPrefsError(msg, "hospital_prefs")

        self.doctor_prefs = converted_doctor_prefs
        self.hospital_prefs = converted_hospital_prefs


    def _convert_caps(self):
        """
        Check the validity of the hospital_caps and convert it into 
        a ndarray.
        """
        try:
            self.hospital_caps = np.array(self.hospital_caps, dtype=int)
        
        except Exception as e:
            msg = f"'hospital_caps' must be a list of non-negative integers.\n" +\
                f"'hospital_caps': {self.hospital_caps}"
            raise InvalidCapsError(msg, "hospital_caps")

        if len(self.hospital_caps) != self.num_hospitals:
            msg = f"The length of 'hospital_caps' must be equal to 'num_hospitals'.\n" +\
                f"'hospital_caps': {self.hospital_caps}"
            raise InvalidCapsError(msg, "hospital_caps")

        if np.any(self.hospital_caps < 0):
            msg = f"'hospital_caps' must be a list of non-negative integers.\n" +\
                f"'hospital_caps': {self.hospital_caps}"
            raise InvalidCapsError(msg, "hospital_caps")


    @staticmethod
    def _convert_prefs_to_ranks(prefs, num_objects):
        num_people = len(prefs)
        outside_option = num_objects
        rank_table = np.full(
            [num_people, num_objects+1], outside_option, dtype=int)

        for p, pref in enumerate(prefs):
            for rank, obj in enumerate(pref):
                rank_table[p, obj] = rank
                if obj == outside_option:
                    break

        return rank_table


    @staticmethod
    def create_setup(
        num_doctors, 
        num_hospitals, 
        outside_score_doctor=0.0, 
        outside_score_hospital=0.0, 
        random_type="normal",
        random_seed=None
        ):
        random_generator = np.random.default_rng(seed=random_seed)
        setup = {}

        if outside_score_doctor in ["min", "max"]:
            prefs = generate_prefs_from_random_scores(
                num_doctors, 
                num_hospitals, 
                outside_score=None, 
                random_type=random_type,
                random_generator=random_generator
            )

            if outside_score_doctor == "min":
                setup["d_prefs"] = prefs
            else:
                setup["d_prefs"] = prefs[:, 0:1]

        else:
            setup["d_prefs"] = generate_prefs_from_random_scores(
                num_doctors, 
                num_hospitals, 
                outside_score_doctor, 
                random_type,
                random_generator
            )

        if outside_score_hospital in ["min", "max"]:
            prefs = generate_prefs_from_random_scores(
                num_hospitals, 
                num_doctors, 
                outside_score=None, 
                random_type=random_type,
                random_generator=random_generator
            )

            if outside_score_hospital == "min":
                setup["h_prefs"] = prefs
            else:
                setup["h_prefs"] = prefs[:, 0:1]

        else:
            setup["h_prefs"] = generate_prefs_from_random_scores(
                num_hospitals, 
                num_doctors,  
                outside_score_hospital, 
                random_type,
                random_generator
            )

        setup["hospital_caps"] = generate_caps_given_sum(
            num_hospitals, int(num_doctors*3/2), random_generator)

        return setup


    def boston(self, doctor_proposing=True):
        """
        Run the Boston algorithm in a many-to-one two-sided matching market.

        By default, this method runs the doctor proposing algorithm.

        Args:
            doctor_proposing : bool, optional
                If True, it runs the doctor proposing alg. Otherwise it 
                runs the hospital proposing alg.

        Returns:
            matching : 1d-ndarray
                List of the matched hospitals. The n-th element indicates 
                the hospital which the n-th doctor matches.
        """
        if not doctor_proposing:
            raise NotImplementedError("Reverse boston is not implemented")

        doctors = list(range(self.num_doctors))
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)
        remaining_caps = np.copy(self.hospital_caps)
        matching = np.full(
            self.num_doctors, 
            self.doctor_outside_option, 
            dtype=int
        )
        next_proposing_rank = 0

        while len(doctors) > 0:
            unmatched_doctors = []
            applied_doctor_ranks = {
                h: [] for h in range(self.num_hospitals)
            }

            for d in doctors:
                h = self.doctor_prefs[d][next_proposing_rank]

                # if d's preference list is exhausted
                if h == self.doctor_outside_option:
                    continue
                
                d_rank = hospital_rank_table[h, d]

                # if d is unacceptable for h
                if d_rank == self.hospital_outside_option:
                    unmatched_doctors.append(d)
                    continue

                applied_doctor_ranks[h].append(d_rank)
                
            for h in range(len(self.hospitals)):
                if len(applied_doctor_ranks[h]) > remaining_caps:
                    applied_doctor_ranks[h].sort()
                
                for d_rank in applied_doctor_ranks[h]:
                    d = self.hospital_prefs[h, d_rank]
                    if remaining_caps[h] > 0:
                        matching[d] = h
                        remaining_caps -= 1
                    else:
                        unmatched_doctors.append(d)


            doctors = unmatched_doctors
            next_proposing_rank += 1

        return matching


    def serial_dictatorship(self, doctor_proposing=True, application_order=None):
        """
        Run the serial dictatorship algorithm in a many-to-one two-sided matching market.

        By default, this method runs the doctor proposing algorithm.

        Args:
            doctor_proposing : bool, optional
                If True, it runs the doctor proposing alg. Otherwise it 
                runs the hospital proposing alg.

            application_order : 1d-array(int), optional
                List of proposing side agents (if doctor_proposing == True, list of doctors).
                If None, then [0, 1, ..., num_doctors-1] is used as application order.

        Returns:
            matching : 1d-ndarray
                List of the matched hospitals. The n-th element indicates 
                the hospital which the n-th doctor matches.
        """
        if not doctor_proposing:
            raise NotImplementedError("Reverse boston is not implemented")

        if application_order is None:
            if doctor_proposing:
                application_order = list(range(self.num_doctors))
            else:
                application_order = list(range(self.num_hospitals))

        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)
        remaining_caps = np.copy(self.hospital_caps)
        matching = np.full(
            self.num_doctors, 
            self.doctor_outside_option, 
            dtype=int
        )

        for d in application_order:
            for rank in range(self.num_hospitals+1):
                h = self.doctor_prefs[d][rank]

                # if d's preference list is exhausted
                if h == self.doctor_outside_option:
                    break
                
                d_rank = hospital_rank_table[h, d]

                # if d is unacceptable for h
                if d_rank == self.hospital_outside_option:
                    pass

                # if cap of h is full
                elif remaining_caps[h] == 0:
                    pass

                else:
                    matching[d] = h
                    remaining_caps[h] -= 1
                    break

        return matching


    def _convert_matching_heap_to_list(self, matched_ranks):
        matching = np.full(
            self.num_doctors, 
            self.doctor_outside_option, 
            dtype=int
        )

        for h, heap in matched_ranks.items():
            for d_rank in heap.values():
                d = self.hospital_prefs[h, d_rank]
                matching[d] = h

        return matching


    def deferred_acceptance_raw_python(self, doctor_proposing=True):
        doctors = list(range(self.num_doctors))
        next_proposing_ranks = np.zeros(self.num_doctors, dtype=int)
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)
        matched_doctor_ranks = {
            h: MaxHeap(self.hospital_caps[h]) for h in range(self.num_hospitals)
        }

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

                # if the doctor's rank is below the outside option
                if d_rank == self.hospital_outside_option:
                    continue

                # if the hospital cap is 0
                if self.hospital_caps[h] == 0:
                    pass

                # if the cap is not full
                elif matched_doctor_ranks[h].length < self.hospital_caps[h]:
                    matched_doctor_ranks[h].push(d_rank)
                    break

                # if the cap is full but a less favorable doctor is matched
                elif d_rank < matched_doctor_ranks[h].root():
                    worst_rank = matched_doctor_ranks[h].replace(d_rank)
                    worst_doctor = self.hospital_prefs[h, worst_rank]
                    doctors.append(worst_doctor)
                    break

        matching = self._convert_matching_heap_to_list(matched_doctor_ranks)
        return matching


    def deferred_acceptance(self, doctor_proposing=True, no_numba=False, new=False):
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
            raise NotImplementedError("Reverse DA is not implemented")

        if no_numba:
            return self.deferred_acceptance_raw_python(doctor_proposing)

        return deferred_acceptance(
            self.num_doctors, 
            self.num_hospitals, 
            self.doctor_prefs, 
            self.hospital_prefs, 
            self.hospital_caps
        )


    def list_blocking_pairs(self, matching):
        doctor_rank_table = self._convert_prefs_to_ranks(
            self.doctor_prefs, self.num_hospitals)
        hospital_rank_table = self._convert_prefs_to_ranks(
            self.hospital_prefs, self.num_doctors)

        # fill current matching rank table
        unmatched_flag = -1
        doctor_matching_ranks = np.empty(self.num_doctors, dtype=int)
        hospital_worst_matching_ranks = np.full(self.num_hospitals, unmatched_flag)
        for d, h in enumerate(matching):
            if h == self.num_hospitals:
                continue
            
            doctor_matching_ranks[d] = doctor_rank_table[d, h]

            if hospital_worst_matching_ranks[h] == unmatched_flag:
                hospital_worst_matching_ranks[h] = hospital_rank_table[h, d]
            elif hospital_worst_matching_ranks[h] < hospital_rank_table[h, d]:
                hospital_worst_matching_ranks[h] = hospital_rank_table[h, d]

        # find blocking pairs
        blocking_pairs = []
        for d, h in enumerate(matching):
            if h != self.num_hospitals:
                if doctor_rank_table[d, h] == self.num_hospitals:
                    blocking_pairs.append((d, self.num_hospitals))

                if hospital_rank_table[h, d] == self.num_doctors:
                    blocking_pairs.append((self.num_doctors, h))

        for d in range(self.num_doctors):
            for h in self.doctor_prefs[d]:
                if (h == self.num_hospitals) or (h == matching[d]):
                    break

                if doctor_rank_table[d, h] < doctor_matching_ranks[d]:
                    # if hospital cap is not full
                    if hospital_worst_matching_ranks[h] == unmatched_flag:
                        if hospital_rank_table[h, d] < self.num_doctors:
                            blocking_pairs.append((d, h))

                    # if hospital is matched with worse doctor
                    elif hospital_rank_table[h, d] < hospital_worst_matching_ranks[h]:
                        blocking_pairs.append((d, h))

        return blocking_pairs


    def get_doctor_matching_ranks(self, matching, doctor_rank_table=None):
        if doctor_rank_table is None:
            doctor_rank_table = self._convert_prefs_to_ranks(
                self.doctor_prefs, self.num_hospitals)
        
        matching_ranks = np.zeros_like(matching)
        for d, h in enumerate(matching):
            matching_ranks[d] = doctor_rank_table[d, h]

        return matching_ranks


    def get_hospital_matching_ranks(self, matching, hospital_rank_table=None):
        if hospital_rank_table is None:
            hospital_rank_table = self._convert_prefs_to_ranks(
                self.hospital_prefs, self.num_doctors)
        
        matching_ranks = [[] for h in range(self.num_hospitals)]
        for d, h in enumerate(matching):
            if h != self.doctor_outside_option:
                matching_ranks[h].append(hospital_rank_table[h, d])

        return matching_ranks


    @staticmethod
    def count_pref_length(prefs, outside_option):
        pref_lengths = []

        for d, li in enumerate(prefs):
            for c, h in enumerate(li):
                if h == outside_option:
                    pref_lengths.append(c)
                    break

            else:
                pref_lengths.append(len(li))

        return pref_lengths


    def analyze_matching(self, matching):
        result = {}

        hospital_matching = {h: [] for h in range(self.num_hospitals+1)}
        for d, h in enumerate(matching):
            hospital_matching[h].append(d)

        result["doctor_pref_lengths"] = self.count_pref_length(
            self.doctor_prefs, 
            self.doctor_outside_option
        )

        result["hospital_pref_lengths"] = self.count_pref_length(
            self.hospital_prefs, 
            self.hospital_outside_option
        )

        result["unmatch_doctor_size"] = len(hospital_matching[self.doctor_outside_option])
        result["matching_size"] = self.num_doctors - result["unmatch_doctor_size"]
        result["unmatch_hospital_cap_size"] = \
            np.sum(self.hospital_caps) - result["matching_size"]
        result["cap_full_hospital_size"] = 0
        for h in range(self.num_hospitals):
            if len(hospital_matching[h]) == self.hospital_caps[h]:
                result["cap_full_hospital_size"] += 1

        matching_ranks = self.get_doctor_matching_ranks(matching)
        result["num_rank1_doctors"] = len(matching_ranks[matching_ranks == 0])
        result["num_rank12_doctors"] = len(matching_ranks[matching_ranks <= 1])

        return result


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
    r1 = m.deferred_acceptance()
    r2 = m.deferred_acceptance(no_numba=True)
    print(r1)
    print(r2)
    
    """

    """
    d_prefs = [
        [4, 0, 5, 1, 2, 3], 
        [1, 4, 0, 5, 2, 3], 
        [2, 0, 5, 1, 3, 4], 
        [3, 0, 5, 1, 2, 4], 
        [0, 1, 5, 2, 3, 4], 
        [0, 2, 5, 1, 3, 4], 
        [0, 2, 3, 5, 1, 4], 
    ]
    h_prefs = [
        [0, 1, 2, 3, 4, 5, 6, 7], 
        [4, 1, 7, 0, 2, 3, 5, 6], 
        [5, 6, 2, 7, 0, 1, 3, 4], 
        [6, 3, 7], 
        [1, 0, 7], 
    ]
    caps = np.array([3, 1, 1, 1, 1])
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

    #"""
    num_doctors, num_hospitals = 3000, 300
    setup = ManyToOneMarket.create_setup(
        num_doctors, 
        num_hospitals, 
        outside_score_doctor=0.99, 
        outside_score_hospital=0.0, 
        random_type="normal",
        random_seed=None
    )

    m = ManyToOneMarket(setup["d_prefs"], setup["h_prefs"], setup["hospital_caps"])
    num_simulation = 10

    import datetime

    start_time = datetime.datetime.now()
    for i in range(num_simulation):
        m.analyze_matching(m.deferred_acceptance())
    
    print(datetime.datetime.now() - start_time)

    start_time = datetime.datetime.now()
    for i in range(num_simulation):
        m.analyze_matching(m.deferred_acceptance(no_numba=True))
    
    print(datetime.datetime.now() - start_time)

    #"""
    
