"""
Utility classes and functions used in this library

"""
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


def to_probability(lst):
    total = sum(lst)
    return [elem / total for elem in lst]


def generate_prefs(num_doctors, num_hospitals, outside_score_doctor=0, outside_score_hospital=0, random_type="normal"):
    """
    ARGS
        num_doctors (int) > 0
        num_hospitals (int) > 0
        outside_score_doctor (bool) : relative "strength" of the outside option
        outside_score_hospital (bool) : relative "strength" of the outside option
        random_type (str) : the probability distribtuion of the score

    RETURN
        d_pref_dict (dict) : key=doctor_id, value=d_pref_list
        h_pref_dict (dict) : key=hospital_id, value=h_pref_list
    """
    # num_doctors = 10
    # num_hospitals = 4
    # outside_score_doctor = 0.1
    # outside_score_hospital = 0.1

    if random_type == "normal":
        # assign scores with normal
        d_scale = 10
        h_scale = 10
        d_score_dict = dict(enumerate(abs(np.random.normal(size=num_doctors, scale=d_scale))))
        h_score_dict = dict(enumerate(abs(np.random.normal(size=num_hospitals, scale=h_scale))))
    elif random_type == "cauchy":
        # assign score with cauchy
        d_score_dict = dict(enumerate(abs(np.random.standard_cauchy(size=num_doctors))))
        h_score_dict = dict(enumerate(abs(np.random.standard_cauchy(size=num_hospitals))))
    elif random_type == "log_normal":
        # assign scores with normal
        d_scale = 1
        h_scale = 1
        d_score_dict = dict(enumerate(np.exp(abs(np.random.normal(size=num_doctors, scale=d_scale)))))
        h_score_dict = dict(enumerate(np.exp(abs(np.random.normal(size=num_hospitals, scale=h_scale)))))
    else:
        raise

    # normalize score
    d_score_sum = sum(d_score_dict.values())
    h_score_sum = sum(h_score_dict.values())
    d_score_dict = dict([(d, score * (1 - outside_score_doctor) / d_score_sum) for d, score in d_score_dict.items()])
    h_score_dict = dict([(h, score * (1 - outside_score_hospital) / h_score_sum) for h, score in h_score_dict.items()])

    if outside_score_hospital > 0:
        d_pref_dict = dict()
        for d in range(num_doctors):
            h_score_dict_copy = h_score_dict.copy()
            top = np.random.choice(list(h_score_dict_copy.keys()), size=1, p=to_probability(h_score_dict_copy.values()))[0]
            h_score_dict_copy.pop(top)
            d_pref = [top] + list(np.random.choice(list(h_score_dict_copy.keys())+[num_hospitals], size=num_hospitals, replace=False, p=to_probability(list(h_score_dict_copy.values())+[outside_score_hospital])))
            d_pref_dict[d] = d_pref
        #d_pref_dict = dict([(d, np.random.choice(num_hospitals+1, size=num_hospitals+1, replace=False, p=list(h_score_dict.values())+[outside_score_hospital])) for d in range(num_doctors)])  # for the case that there are doctors who want to go anywhere
        #d_pref_dict = dict([(d, np.random.choice(num_hospitals+1, size=num_hospitals+1, replace=False)) for d in range(num_doctors)])  # for uniform
    elif outside_score_hospital == 0:
        d_pref_dict = dict([(d, np.random.choice(num_hospitals, size=num_hospitals, replace=False, p=list(h_score_dict.values())).tolist()) for d in range(num_doctors)])
        #d_pref_dict = dict([(d, np.random.choice(num_hospitals, size=num_hospitals, replace=False)) for d in range(num_doctors)])  # for uniform
    else:
        raise

    if outside_score_doctor > 0:
        h_pref_dict = dict()
        for h in range(num_hospitals):
            d_score_dict_copy = d_score_dict.copy()
            top = np.random.choice(list(d_score_dict_copy.keys()), size=1, p=to_probability(d_score_dict_copy.values()))[0]
            d_score_dict_copy.pop(top)
            h_pref = [top] + list(np.random.choice(list(d_score_dict_copy.keys())+[num_doctors], size=num_doctors, replace=False, p=to_probability(list(d_score_dict_copy.values())+[outside_score_doctor])))
            h_pref_dict[h] = h_pref
        #h_pref_dict = dict([(h, np.random.choice(num_doctors+1, size=num_doctors+1, replace=False, p=list(d_score_dict.values())+[outside_score_doctor])) for h in range(num_hospitals)])  # for the case that there are hospitals who want to hire nobody
        #h_pref_dict = dict([(h, np.random.choice(num_doctors+1, size=num_doctors+1, replace=False)) for h in range(num_hospitals)])  # for uniform
    elif outside_score_doctor == 0:
        h_pref_dict = dict([(h, np.random.choice(num_doctors, size=num_doctors, replace=False, p=list(d_score_dict.values())).tolist()) for h in range(num_hospitals)])
        #h_pref_dict = dict([(h, np.random.choice(num_doctors, size=num_doctors, replace=False)) for h in range(num_hospitals)])  # for uniform
    else:
        raise

    return d_pref_dict, h_pref_dict


if __name__ == "__main__":
    num_doctors = 56
    num_hospitals = 80
    outside_score_doctor = 0
    outside_score_hospital = 0  #15*1 / num_doctors
    random_type = "cauchy"

    d_pref_dict, h_pref_dict = generate_prefs(num_doctors, num_hospitals, outside_score_doctor, outside_score_hospital, random_type)

    d_pref_df = pd.DataFrame(d_pref_dict.values())
    df_flag = pd.DataFrame(np.zeros_like(d_pref_df))
    df_flag[d_pref_df == num_hospitals] = 1
    df_flag = df_flag.cumsum(axis=1)
    df_flag[df_flag == 1] = np.nan
    df_flag += 1
    d_pref_df = (d_pref_df * df_flag).astype("Int64", errors="ignore").iloc[:, :-1]
    d_pref_df.index = [f"{num+1}さん" for num in d_pref_df.index]
    d_pref_df.columns = [f"希望{num+1}" for num in d_pref_df.columns]
    d_pref_df = d_pref_df.applymap(lambda x: x if pd.isna(x) else f"部署{x+1}")
    #d_pref_df.to_csv("d_pref.csv", encoding="shiftjis")

    h_pref_df = pd.DataFrame(h_pref_dict.values())
    df_flag = pd.DataFrame(np.zeros_like(h_pref_df))
    df_flag[h_pref_df == num_doctors] = 1
    df_flag = df_flag.cumsum(axis=1)
    df_flag[df_flag == 1] = np.nan
    df_flag += 1
    h_pref_df = (h_pref_df * df_flag).astype("Int64", errors="ignore").iloc[:, :-1]
    h_pref_df.index = [f"部署{num+1}" for num in h_pref_df.index]
    h_pref_df.columns = [f"希望{num+1}" for num in h_pref_df.columns]
    h_pref_df = h_pref_df.applymap(lambda x: x if pd.isna(x) else f"{x+1}さん")
    #h_pref_df.to_csv("h_pref.csv", encoding="shiftjis")


    #pd.Series([elem[0] for elem in d_pref_dict.values()]).value_counts().plot(kind="bar")
    #pd.Series([elem[0] for elem in h_pref_dict.values()]).value_counts().plot(kind="bar", title="1位指名の数", xlabel="医者ID")

    #pd.Series(np.where(pd.DataFrame(d_pref_dict.values()).values == num_hospitals)[1]).value_counts()
    #pd.Series(np.where(pd.DataFrame(d_pref_dict.values()).values == num_hospitals)[1]).median()
    #pd.Series(np.where(pd.DataFrame(h_pref_dict.values()).values == num_doctors)[1]).median()
