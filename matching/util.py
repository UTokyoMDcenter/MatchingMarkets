"""
Utility classes and functions used in this library

"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def generate_prefs(num_doctors, num_hospitals, outside_score_doctor=0, outside_score_hospital=0):
    """
    ARGS
        num_doctors (int) > 0
        num_hospitals (int) > 0
        outside_score_doctor (bool) : relative "strength" of the outside option
        outside_score_hospital (bool) : relative "strength" of the outside option

    RETURN
        d_pref_dict (dict) : key=doctor_id, value=d_pref_list
        h_pref_dict (dict) : key=hospital_id, value=h_pref_list
    """
    # num_doctors = 10
    # num_hospitals = 4
    # outside_score_doctor = 0.1
    # outside_score_hospital = 0.1

    # assign scores with normal
    #d_scale = 10
    #h_scale = 10
    #d_score_dict = dict(enumerate(abs(np.random.normal(size=num_doctors, scale=d_scale))))
    #h_score_dict = dict(enumerate(abs(np.random.normal(size=num_hospitals, scale=h_scale))))

    # assign score with cauchy
    d_score_dict = dict(enumerate(abs(np.random.standard_cauchy(size=num_doctors))))
    h_score_dict = dict(enumerate(abs(np.random.standard_cauchy(size=num_hospitals))))

    # normalize score
    d_score_sum = sum(d_score_dict.values())
    h_score_sum = sum(h_score_dict.values())
    d_score_dict = dict([(d, score * (1 - outside_score_doctor) / d_score_sum) for d, score in d_score_dict.items()])
    h_score_dict = dict([(h, score * (1 - outside_score_hospital) / h_score_sum) for h, score in h_score_dict.items()])

    if outside_score_doctor > 0:
        d_pref_dict = dict([(d, np.random.choice(num_hospitals+1, size=num_hospitals+1, replace=False, p=list(h_score_dict.values())+[outside_score_hospital])) for d in range(num_doctors)])
        #d_pref_dict = dict([(d, np.random.choice(num_hospitals+1, size=num_hospitals+1, replace=False)) for d in range(num_doctors)])  # for uniform
    elif outside_score_doctor == 0:
        d_pref_dict = dict([(d, np.random.choice(num_hospitals, size=num_hospitals, replace=False, p=list(h_score_dict.values()))) for d in range(num_doctors)])
        #d_pref_dict = dict([(d, np.random.choice(num_hospitals, size=num_hospitals, replace=False)) for d in range(num_doctors)])  # for uniform
    else:
        raise

    if outside_score_hospital > 0:
        h_pref_dict = dict([(h, np.random.choice(num_doctors+1, size=num_doctors+1, replace=False, p=list(d_score_dict.values())+[outside_score_doctor])) for h in range(num_hospitals)])
        #h_pref_dict = dict([(h, np.random.choice(num_doctors+1, size=num_doctors+1, replace=False)) for h in range(num_hospitals)])  # for uniform
    elif outside_score_hospital == 0:
        h_pref_dict = dict([(h, np.random.choice(num_doctors, size=num_doctors, replace=False, p=list(d_score_dict.values()))) for h in range(num_hospitals)])
        #h_pref_dict = dict([(h, np.random.choice(num_doctors, size=num_doctors, replace=False)) for h in range(num_hospitals)])  # for uniform
    else:
        raise

    return d_pref_dict, h_pref_dict


if __name__ == "__main__":
    num_doctors = 56
    num_hospitals = 80
    outside_score_doctor = 1 / num_hospitals
    outside_score_hospital = 1 / num_doctors
    d_pref_dict, h_pref_dict = generate_prefs(num_doctors, num_hospitals, outside_score_doctor, outside_score_hospital)
    pd.Series([elem[0] for elem in d_pref_dict.values()]).value_counts().plot(kind="bar")
    pd.Series([elem[0] for elem in h_pref_dict.values()]).value_counts().plot(kind="bar")
