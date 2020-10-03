import numpy as np

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

    school_consts
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
        #self._check_caps()
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
            if D in self.school_consts[s]:
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

"""
num_students = len(student_prefs)
num_schools = len(school_prefs)
student_outside_option = num_schools
school_outside_option = num_students
school_rank_table = _convert_prefs_to_ranks(school_prefs, num_students)
student_rank_table = _convert_prefs_to_ranks(student_prefs, num_schools)
p = np.array([1, 1, 1])
s = 0
demanding_student_list = []
for i in school_prefs[s][:num_students - p[s] + 1]:  # i is above or on the cutoff line of s
    # i = 0
    rank_s = student_rank_table[i][s]  # rank of s for i
    if rank_s != student_outside_option:  # s is acceptable for i
        if all([school_rank_table[s_prime][i] > num_students - p[s_prime] for s_prime in student_prefs[i][:rank_s]]):  # If s_prime is above s for i, i is below the cutoff line of s_prime
            demanding_student_list.append(i)
"""

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
        [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {1, 2}, {2, 3},],
        [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3},],
        [{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}, {0, 1, 2}, {0, 1, 3}],
    ]
    m = ManyToOneMarketWithConstraints(student_prefs, school_prefs, school_consts)
    m.cutoff_adjustment_algorithm()
