"""
Basic two sided matching markets.

"""
import numpy as np
import pandas as pd
import warnings
from MatchingMarkets.util import InvalidPrefsError, InvalidCapsError


class Parser(object):
    def __init__(self):
        pass


    def parse_prefs(self, df, pref_cols, unique_names=None):
        self.name_to_index, self.index_to_name = {}, {}
        if unique_names is None:
            unique_names = pd.unique(df[pref_cols].to_numpy().flatten())
        else:
            original_unique_name = unique_names
            unique_names = pd.unique(original_unique_name)
            if len(unique_names) != len(original_unique_name):
                warnings.warn(f"`unique_names` has duplicates.")

        for i, name in enumerate(unique_names):
            self.name_to_index[name] = i
            self.index_to_name[i] = name

        self.unique_names = unique_names
        self.outside_option = len(self.unique_names)

        num_agents, num_objects = df.shape[0], len(self.unique_names) + 1
        prefs = np.full((num_agents, num_objects), fill_value=self.outside_option)

        for i, c in enumerate(pref_cols):
            prefs[:, i] = df[c].map(self.name_to_index).fillna(self.outside_option)

        return prefs


if __name__ == "__main__":
    pass