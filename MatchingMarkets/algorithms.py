import numba
import numba.pycc
import numpy as np
from util import MaxHeap

cc = numba.pycc.CC("matching_alg")


@numba.jit(nopython=True)
def convert_prefs_to_ranks(prefs, num_objects):
    num_people = len(prefs)
    outside_option = num_objects
    rank_table = np.full(
        (num_people, num_objects+1), 
        outside_option, 
        dtype=numba.int64
    )

    for p in range(len(prefs)):
        pref = prefs[p]
        for rank, obj in enumerate(pref):
            rank_table[p, obj] = rank
            if obj == outside_option:
                break

    return rank_table


@numba.jit(nopython=True)
def convert_matching_heap_to_list(num_doctors, num_hospitals, hospital_prefs, matched_ranks):
    doctor_outside_option = num_hospitals
    matching = np.full(
        num_doctors, 
        doctor_outside_option, 
        dtype=numba.i8
    )

    for h, heap in enumerate(matched_ranks):
        for d_rank in heap.values():
            d = hospital_prefs[h, d_rank]
            matching[d] = h

    return matching


@cc.export("deferred_acceptance", "i8[:](i8, i8, i8[:, :], i8[:, :], i8[:])")
def deferred_acceptance(num_doctors, num_hospitals, doctor_prefs, hospital_prefs, hospital_caps):
    doctor_outside_option = num_hospitals
    hospital_outside_option = num_doctors
    doctors = list(range(num_doctors))
    next_proposing_ranks = np.zeros(num_doctors, dtype=numba.i8)
    hospital_rank_table = convert_prefs_to_ranks(hospital_prefs, num_doctors)
    matched_doctor_ranks = [MaxHeap(hospital_caps[h]) for h in range(num_hospitals)]

    while len(doctors) > 0:
        d = doctors.pop()
        first_rank = next_proposing_ranks[d]
        d_pref = doctor_prefs[d]

        for rank in range(first_rank, num_hospitals+1):
            next_proposing_ranks[d] += 1
            h = d_pref[rank]

            # if this doctor's preference list is exhausted
            if h == doctor_outside_option:
                break

            d_rank = hospital_rank_table[h, d]

            # if the doctor's rank is below the outside option
            if d_rank == hospital_outside_option:
                continue

            # if the hospital cap is 0
            if hospital_caps[h] == 0:
                continue

            # if the cap is not full
            if hospital_caps[h] > matched_doctor_ranks[h].length:
                matched_doctor_ranks[h].push(d_rank)
                break

            # if the cap is full but a less favorable doctor is matched
            elif d_rank < matched_doctor_ranks[h].root():
                worst_rank = matched_doctor_ranks[h].replace(d_rank)
                worst_doctor = hospital_prefs[h, worst_rank]
                doctors.append(worst_doctor)
                break

    matching = convert_matching_heap_to_list(
        num_doctors, 
        num_hospitals, 
        hospital_prefs,
        matched_doctor_ranks
    )

    return matching


def compile():
    cc.compile()


if __name__ == "__main__":
    cc.compile()
