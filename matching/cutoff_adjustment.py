"""
TODO : To try more complicated criteria of feasibility. Consider ages of students and impose constraints about the sum of ages on schools
"""

import numpy as np
import heapq
from itertools import permutations
from util import InvalidPrefsError, InvalidCapsError


class ManyToOneMarketWithConstraints(object):
    """
    Basic class for the model of a many-to-one two-sided matching market.

    Attributes
    ----------
    num_students : int
        The number of students.

    num_schools : int
        The number of schools.

    student_prefs : 2d-array(int)
        The list of students' preferences over the schools and the outside option.
        The elements must be 0 <= x <= num_schools.
        The number `num_schools` is considered as an outside option.

    school_prefs : 2d-array(int)
        The list of school' preferences over the students and the outside option.
        The elements must be 0 <= x <= num_students.
        The number `num_students` is considered as an outside option.

    school_consts : 1d-array(function)
        The list of functions that take a set of students and judge whether it is feasible for school.
    """
    def __init__(self, student_prefs, school_prefs, school_consts):
        self.num_students = len(student_prefs)
        self.num_schools = len(school_prefs)
        self.student_prefs = student_prefs
        self.school_prefs = school_prefs
        self.student_outside_option = self.num_schools
        self.school_outside_option = self.num_students
        self.school_consts = school_consts
        self._check_prefs()
        self._check_consts()
        self.school_rank_table = self._convert_prefs_to_ranks(self.school_prefs, self.num_students)
        self.student_rank_table = self._convert_prefs_to_ranks(self.student_prefs, self.num_schools)


    def _check_prefs(self):
        """
        Check the validity of preferences.
        """
        try:
            self.student_prefs = np.array(self.student_prefs, dtype=int)
            self.school_prefs = np.array(self.school_prefs, dtype=int)

        except Exception as e:
            msg = "Each pref must be a matrix of integers.\n" +\
                f"'student_prefs': {self.student_prefs}\n" +\
                f"'school_prefs': {self.school_prefs}"
            raise InvalidPrefsError(msg)

        if np.min(self.student_prefs) < 0 or \
            np.max(self.student_prefs) > self.student_outside_option:
            msg = \
                "Elements of 'student_prefs' must be 0 <= x <= 'num_schools'.\n" +\
                f"'student_prefs': {self.student_prefs}"
            raise InvalidPrefsError(msg)

        if np.min(self.school_prefs) < 0 or \
            np.max(self.school_prefs) > self.school_outside_option:
            msg = \
                "Elements of 'school_prefs' must be 0 <= x <= 'num_students'\n" +\
                f"'school_prefs': {self.school_prefs}"
            raise InvalidPrefsError(msg)


    def _check_consts(self):
        assert len(self.school_consts) == self.num_schools, "Indicate constraints for all schools."
        #assert all([all([students <= set(range(self.num_students)) for students in feasible_sets]) for feasible_sets in self.school_consts]), "Feasible sets should be a subset of the power set of the set of students."


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


    def demand_profile(self, s, p):
        """
        Args
        -------------
            s : int
                The index of school. 0 <= s < num_schools should hold.
            p : (num_schools)-d array
                The cutoff profile. 1 <= p[i] <= num_students should hold for all i.
        Returns
        -------------
            A set of integers.
        """
        # p = np.array([2, 3, 1])
        # s = 1
        demanding_student_list = []
        for i in self.school_prefs[s][:self.num_students - p[s] + 1]:  # i is above or on the cutoff line of s
            rank_s = self.student_rank_table[i][s]  # rank of s for i
            if rank_s != self.student_outside_option:  # s is acceptable for i
                if all([self.school_rank_table[s_prime][i] > self.num_students - p[s_prime] for s_prime in self.student_prefs[i][:rank_s]]):  # If s_prime is above s for i, i is below the cutoff line of s_prime
                    demanding_student_list.append(i)
        return set(demanding_student_list)


    def cutoff_adjustment(self, p):
        """
            p : (num_schools)-d array
                The cutoff profile. 1 <= p[i] <= num_students should hold for all i.
        """
        T = []
        for s in range(self.num_schools):
            D = self.demand_profile(s, p)
            if self.school_consts[s](D):
                T.append(p[s])
            else:
                T.append(p[s] + 1)
        return np.array(T)


    def cutoff_adjustment_algorithm(self):
        # Find the fixed point of T.
        prev_p = np.ones(self.num_schools, dtype=int)
        res_p, curr_p = None, None
        while True:
            curr_p = self.cutoff_adjustment(prev_p)
            if all(curr_p == prev_p):
                res_p = curr_p
                break
            prev_p = curr_p

        # Matching for schools
        matching_school = [self.demand_profile(school, res_p) for school in range(self.num_schools)]

        # Matching for students
        matching_student_dict = dict()
        for s, mu_s in enumerate(matching_school):
            for i in mu_s:
                matching_student_dict[i] = s
        matching_student = [elem[1] for elem in sorted(list(matching_student_dict.items()))]

        return matching_student


    def is_feasible(self, matching_student):
        # matching_student = [0, 1, 0, 2]
        for s in range(self.num_schools):
            # s = 0
            # It is easier if this function takes matching_school as its argument.
            if not self.school_consts[s](set(np.where(np.array(matching_student) == s)[0])):
                return False
        return True


    def is_indivisually_rational(self, matching_student):
        for i in range(self.num_students):
            # Assume that unacceptable schools are not in student_prefs[i].
            if not matching_student[i] in self.student_prefs[i]:
                return False
        return True


    def is_fair(self, matching_student):
        for i, i_prime in permutations(range(self.num_students), 2):
            for s in range(self.num_schools):
                # The following has some computational redundancy.
                cond_1 = self.student_rank_table[i][s] < self.student_rank_table[i][matching_student[i]]  # s is better for i than the school i matches.
                cond_2 = matching_student[i_prime] == s  # i_prime matches s
                cond_3 = self.school_rank_table[s][i] < self.school_rank_table[s][i_prime]  # i is better than i_prime for s
                if all([cond_1, cond_2, cond_3]):
                    return False
        return True


if __name__ == "__main__":
    student_prefs = [
        [0, 2, 1],
        [1, 0, 2],
        [0, 1, 2],
        [2, 0, 1],
    ]
    school_prefs = [
        [3, 2, 1, 0],
        [1, 0, 2, 3],
        [2, 1, 3, 0],
    ]
    school_consts = [
        lambda students: students in [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {1, 2}, {2, 3},],
        lambda students: students in [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},],
        lambda students: students in [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}],
    ]
    school_consts = [
        lambda students: students in [{0}, {1}, {2}, {3},],
        lambda students: students in [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},],
        lambda students: students in [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}],
    ]
    m = ManyToOneMarketWithConstraints(student_prefs, school_prefs, school_consts)
    matching_student = m.cutoff_adjustment_algorithm()
    m.is_feasible(matching_student)
    m.is_indivisually_rational(matching_student)
    m.is_fair(matching_student)
