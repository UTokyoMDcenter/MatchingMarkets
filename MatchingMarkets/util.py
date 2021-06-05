"""
Utility classes and functions used in this library

"""
import numpy as np
import heapq
import numba
import numba.experimental


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


heap_spec = [
    ("length", numba.i8), 
    ("arr_size", numba.i8), 
    ("array", numba.i8[:]), 
]


@numba.experimental.jitclass(heap_spec)
class MaxHeap(object):
    def __init__(self, arr_size):
        self.length = 0
        self.arr_size = arr_size
        self.array = np.empty(arr_size, dtype=numba.i8)


    @staticmethod
    def _comp(elem1, elem2):
        return elem1 >= elem2


    def _swap(self, ind1, ind2):
        temp = self.array[ind1]
        self.array[ind1] = self.array[ind2]
        self.array[ind2] = temp


    def _shiftup(self, index):
        while index > 0:
            parent = (index-1) // 2
            if self._comp(self.array[parent], self.array[index]):
                break

            self._swap(parent, index)
            index = parent


    def _shiftdown(self):
        index = 0
        left_child, right_child = 1, 2
        while left_child < self.length:
            if right_child < self.length:
                if self._comp(self.array[left_child], self.array[right_child]):
                    larger_child = left_child
                else:
                    larger_child = right_child
            else:
                larger_child = left_child

            if self._comp(self.array[index], self.array[larger_child]):
                break

            self._swap(index, larger_child)

            index = larger_child
            left_child = 2 * index + 1
            right_child = 2 * index + 2


    def push(self, value):
        if self.length == self.arr_size:
            raise IndexError(
                "The heap is full (its length already reaches `arr_size`).")

        self.array[self.length] = value
        self.length += 1
        self._shiftup(self.length-1)


    def pop(self):
        if self.length == 0:
            raise IndexError("The heap is empty.")

        self.length -= 1
        elem = self.array[0]
        self.array[0] = self.array[self.length]
        self._shiftdown()
        return elem


    def replace(self, value):
        elem = self.array[0]
        self.array[0] = value
        self._shiftdown()
        return elem


    def values(self):
        return self.array[:self.length]


    def root(self):
        if self.length == 0:
            raise IndexError("The heap is empty.")
        
        return self.array[0]


    def is_full(self):
        return self.length == self.arr_size


@numba.experimental.jitclass(heap_spec)
class MinHeap(object):
    """
    Currently inheritance from a numba.jitclass is not supported.
    """
    def __init__(self, arr_size):
        self.length = 0
        self.arr_size = arr_size
        self.array = np.empty(arr_size, dtype=numba.i8)


    @staticmethod
    def _comp(elem1, elem2):
        return elem1 <= elem2


    def _swap(self, ind1, ind2):
        temp = self.array[ind1]
        self.array[ind1] = self.array[ind2]
        self.array[ind2] = temp


    def _shiftup(self, index):
        while index > 0:
            parent = (index-1) // 2
            if self._comp(self.array[parent], self.array[index]):
                break

            self._swap(parent, index)
            index = parent


    def _shiftdown(self):
        index = 0
        left_child, right_child = 1, 2
        while left_child < self.length:
            if right_child < self.length:
                if self._comp(self.array[left_child], self.array[right_child]):
                    larger_child = left_child
                else:
                    larger_child = right_child
            else:
                larger_child = left_child

            if self._comp(self.array[index], self.array[larger_child]):
                break

            self._swap(index, larger_child)

            index = larger_child
            left_child = 2 * index + 1
            right_child = 2 * index + 2


    def push(self, value):
        if self.length == self.arr_size:
            raise IndexError(
                "The heap is full (its length already reaches `arr_size`).")

        self.array[self.length] = value
        self.length += 1
        self._shiftup(self.length-1)


    def pop(self):
        if self.length == 0:
            raise IndexError("The heap is empty.")

        self.length -= 1
        elem = self.array[0]
        self.array[0] = self.array[self.length]
        self._shiftdown()
        return elem


    def replace(self, value):
        elem = self.array[0]
        self.array[0] = value
        self._shiftdown()
        return elem


    def values(self):
        return self.array[:self.length]


    def root(self):
        if self.length == 0:
            raise IndexError("The heap is empty.")
        
        return self.array[0]


    def is_full(self):
        return self.length == self.arr_size


def shuffle_each_row_prev(
    arr, 
    random_generator, 
    outside_option=None, 
    allow_op_first=False
    ):
    x, y = arr.shape
    rows = np.indices((x, y))[0]
    cols = [random_generator.permutation(y) for _ in range(x)]
    
    if (outside_option is not None) and (not allow_op_first):
        while True: 
            invalid_rows = np.where(arr[rows, cols][:, 0] == outside_option)[0]
            if len(invalid_rows) == 0:
                break
            
            new_cols = [random_generator.permutation(y) for _ in range(x)]
            for r in invalid_rows:
                cols[r] = new_cols[r]

    return arr[rows, cols]


def to_probability(li):
    return li / np.sum(li)


def shuffle_list(
    li, 
    size=1, 
    probs=None,
    outside_option=None,
    random_generator=None
    ):
    """
    Args:
        li : 1d array-like(int)
            The list to be shuffled.
        
        size : int, optional
            The sample size of shuffle trials.
        
        probs : 1d array-like(float), optional
            The probability each element of `li` is drawn. The size of `probs` 
            should be same as that of `li`. Each element should be >= 0. 
            If None, then probs will be uniform over the list.
        
        outside_option : int or None, optional
            An integer that is in `li`. If not None, then the value will never 
            be at the beginning of the shuffled list.

        random_generator : numpy.random.Generator, optional
            The random generator. If None, then a generator is initialized in
            this function.  

    Return:
        shuffled_lists : 2d-array(int)
            The list of shuffled lists. shape=(size, len(li)).
    """
    li = np.array(li)
    list_size = len(li)
    indexes = np.arange(list_size)
    shuffled_lists = np.empty(shape=(size, list_size), dtype=int)

    if random_generator is None:
        random_generator = np.random.default_rng()

    if probs is None:
        probs = np.ones(list_size) / list_size
    
    else:
        probs = np.array(probs)

        if len(probs) != list_size:
            raise ValueError(f"The size of `li` and `probs` must be the same.")

        if np.sum(probs <= 0) > 0:
            raise ValueError(f"Elements of `probs` must be strictly greter than 0.")

    if outside_option is None:
        # If outside_option is not specified, simply shuffle the list.
        probs = to_probability(probs)
        for i in range(size):
            shuffled_lists[i, :] = random_generator.choice(
                indexes, 
                size=list_size, 
                replace=False, 
                p=probs
            )
    else:
        # If outside_option is specified, then 
        # 1. randomly choose the top elements from the list except 
        # the outside option, 
        # 2. randomly shuffle the remaining elements and the outside option.
        op_indexes = indexes[li == outside_option]
        if len(op_indexes) == 0:
            raise ValueError(f"`outside_option`: {outside_option} is not in `li`.")

        op_index = op_indexes[0]

        probs_without_op = np.copy(probs)
        probs_without_op[op_index] = 0
        probs_without_op = to_probability(probs_without_op)

        for i in range(size):
            shuffled_lists[i, 0] = random_generator.choice(
                indexes, 
                size=1, 
                replace=False, 
                p=probs_without_op
            )

        for i in range(size):
            probs_remaining = np.copy(probs)
            probs_remaining[shuffled_lists[i, 0]] = 0
            probs_remaining = to_probability(probs_remaining)

            shuffled_lists[i, 1:] = random_generator.choice(
                indexes, 
                size=list_size-1, 
                replace=False, 
                p=probs_remaining
            )
        

    return li[shuffled_lists]


def generate_random_prefs(
    num_agents, 
    num_objects, 
    outside_option=False, 
    random_generator=None
    ):
    """
    Randomly generate preference lists of agents.

    """
    if random_generator is None:
        random_generator = np.random.default_rng()

    if outside_option:
        len_pref = num_objects + 1
        op = num_objects
    else:
        len_pref = num_objects
        op = None

    prefs = shuffle_list(
        np.arange(len_pref), 
        size=num_agents, 
        probs=None,
        outside_option=op,
        random_generator=random_generator
    )

    return prefs


def generate_prefs_from_scores(
    num_agents, 
    num_objects, 
    scores, 
    outside_score=None,
    random_generator=None
    ):
    if random_generator is None:
        random_generator = np.random.default_rng()
    
    if type(scores) is np.ndarray:
        scores = scores.tolist()
    
    if outside_score is not None:
        scores.append(outside_score)

    # normalize score (logit)
    probs = to_probability(np.exp(scores))

    if outside_score is None:
        prefs = shuffle_list(
            np.arange(num_objects), 
            size=num_agents, 
            probs=probs,
            outside_option=None,
            random_generator=random_generator
        )

    else:
        prefs = shuffle_list(
            np.arange(num_objects+1), 
            size=num_agents, 
            probs=probs,
            outside_option=num_objects,
            random_generator=random_generator
        )

    return prefs


def generate_prefs_from_random_scores(
    num_agents, 
    num_objects, 
    outside_score=None, 
    random_type="normal",
    random_generator=None
    ):
    """
    Args:
        num_agents : int(>0)
            The length of preference lists.
        
        num_objects : int(>0)
            The size of objects over which each agent's preference is defined.
        
        outside_score : float(0<=x<=1) or None
            Relative "strength" of the outside option. If None is set, 
            then outside option will not be included in the preferences.
        
        random_type : str, optional. In ['normal', 'cauchy', 'lognormal']
            The probability distribtuion of the score.

        random_generator : numpy.random.Generator, optional
            The random generator. If None, then a generator is initialized in
            this function. 
    
    Return:
        prefs : 2d-array(int)
            The list of agents' preferences over the objects and the outside option.
            The elements must be 0 <= x <= num_objects. 
            The number `num_objects` is considered as an outside option.
    """
    if random_generator is None:
        random_generator = np.random.default_rng()

    if outside_score is not None:
        if outside_score < 0 or 1 < outside_score:
            raise ValueError(f"`outside_score` must be 0 <= x <= 1")

    adjusted_op_score = None

    if random_type == "normal":
        # assign scores with normal
        scale = 1.0
        scores = random_generator.normal(size=num_objects, scale=scale)

        if outside_score is not None:
            # convert [0, 1] -> [-3\sigma, 3\sigma]
            adjusted_op_score = (outside_score - 0.5) * (3 * scale / 0.5)
    
    elif random_type == "cauchy":
        # assign scores with cauchy
        scores = random_generator.standard_cauchy(size=num_objects)

        if outside_score is not None:
            # convert [0, 1] -> [-3, 3]
            adjusted_op_score = (outside_score - 0.5) * (3 / 0.5)
    
    elif random_type == "lognormal":
        # assign scores with log normal
        sigma = 1.0
        scores = random_generator.lognormal(size=num_objects, sigma=sigma)

        if outside_score is not None:
            # convert [0, 1] -> [-3\sigma, 3\sigma]
            mean = np.exp(np.power(sigma, 2) / 2)
            std = np.sqrt(np.exp(np.power(sigma, 2)) * (np.exp(np.power(sigma, 2)) - 1))
            adjusted_op_score = mean + (outside_score - 0.5) * (3 * std / 0.5)
    
    else:
        raise ValueError("`random_type` must be in ['normal', 'cauchy', 'lognormal'].")

    prefs = generate_prefs_from_scores(
        num_agents, 
        num_objects, 
        scores, 
        outside_score=adjusted_op_score,
        random_generator=random_generator
    )

    return prefs


def round_caps_to_meet_sum(li, target_sum, random_generator=None):
    li = np.array(li)
    total = np.sum(li)

    if total <= target_sum:
        return li

    base_li = li * target_sum / total
    rounded_li = np.floor(base_li)
    rounded_total = np.sum(rounded_li)

    # For breaking ties
    temp = np.empty([2, len(li)], 
        dtype=[("value", float), ("breaking_tie_order", int)])
    temp["value"] = -1 * (base_li - rounded_li)
    
    if random_generator is None:
        temp["breaking_tie_order"] = np.arange(len(li))
    else:
        order = np.arange(len(li))
        random_generator.shuffle(order)
        temp["breaking_tie_order"] = order
    
    surplus_ordered_indices = np.argsort(
        temp, order=["value", "breaking_tie_order"])[0]
    rounded_li[surplus_ordered_indices[0:int(target_sum-rounded_total)]] += 1
    return rounded_li.astype(int)


def generate_caps_given_sum(len_list, target_sum, random_generator=None):
    if random_generator is None:
        random_generator = np.random.default_rng()

    # If target_sum > len_list, then set min(caps) == 1.
    original_target_sum = target_sum
    if target_sum > len_list:
        target_sum -= len_list

    caps = random_generator.gamma(
        shape=np.sqrt(target_sum), 
        scale=np.sqrt(target_sum), 
        size=len_list
    )
    caps = np.round(caps).astype(int)
    caps = round_caps_to_meet_sum(caps, target_sum)

    if original_target_sum > len_list:
        caps += 1

    return caps


if __name__ == "__main__":
    pass
