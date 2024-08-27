"""
Basic one-sided matching markets.

"""
import numpy as np
import networkx as nx
from MatchingMarkets.util import InvalidPrefsError, InvalidCapsError, MaxHeap, \
    MarketBase


class ObjectMarket(MarketBase):
    """
    Basic class for the model of a one-sided matching market.

    Attributes
    ----------
    num_agents : int
        The number of agents.
    
    num_objects : int
        The number of objects.
    
    agent_prefs : 2d-array(int)
        The list of agents' preference list over the objects and the outside option.
        The elements must be 0 <= x <= num_objects. 
        The number `num_objects` is considered as an outside option.

    object_caps : 1d-array(int, optional)
        The list of the capacities(quantities) of the objects. The elements must 
        be non-negative. If nothing is specified, then all caps are set to be 1.
    """
    def __init__(self, num_agents, num_objects, agent_prefs, object_caps=None):
        self.num_agents = num_agents
        self.num_objects = num_objects
        self.outside_option = num_objects
        self.agent_prefs = np.array(agent_prefs, dtype=int)

        if object_caps is None:
            self.object_caps = np.ones(num_objects, dtype=int)
        else:
            self.object_caps = np.array(object_caps, dtype=int)


    def serial_dictatorship(self, application_order=None):
        """
        Run the serial dictatorship algorithm in a one-sided matching market.

        Args:
            application_order : 1d-array(int), optional
                List of agents (the first agent gets an object first).
                If None, then [0, 1, ..., num_agents-1] is used as application order.

        Returns:
            matching : 1d-ndarray
                List of the matched objects. The n-th element indicates 
                the object which the n-th agent matches.
        """
        if application_order is None:
            application_order = list(range(self.num_agents))

        remaining_caps = np.copy(self.object_caps)
        matching = np.full(
            self.num_agents, 
            self.outside_option, 
            dtype=int
        )

        for a in application_order:
            for obj in self.agent_prefs[a]:
                # if a's preference list is exhausted
                if obj == self.outside_option:
                    break
                
                # if cap of h is full
                if remaining_caps[obj] == 0:
                    pass

                else:
                    matching[a] = obj
                    remaining_caps[obj] -= 1
                    break

        return matching


    def YRMH_IGYT(self, initial_assignment, application_order=None):
        """
        Run the YRMH-IGYT(You request my house - I get your turn) algorithm 
        in a one-sided matching market.

        Args:
            initial_assignment : 1d-array(int)
                List of the initial assignments of the agents.
                The elements must be 0 <= x <= `num_objects` where
                `num_objects` stands for an unmatch.

        Returns:
            matching : 1d-ndarray
                List of the matched objects. The n-th element indicates 
                the object that the n-th agent matches.
        """
        if application_order is None:
            application_order = list(range(self.num_agents))
        
        matching = np.full(
            self.num_agents, 
            fill_value=self.outside_option,
            dtype=int
        )
        next_proposing_ranks = np.zeros(self.num_agents, dtype=int)
        current_init_assignment = initial_assignment.copy()
        remaining_caps = np.copy(self.object_caps)

        for priority_a in application_order:
            if matching[priority_a] != self.outside_option:
                continue
            
            edges = []

            # add edges from agents to objects
            for a, pref in enumerate(self.agent_prefs):
                # if a is matched, skip
                if matching[a] != self.outside_option:
                    continue

                next_rank = next_proposing_ranks[a]

                # point to the most favorite object whose cap is not full
                for obj in pref[next_rank:]:
                    # if a's preference is exausted, remove a from the market
                    if obj == self.outside_option:
                        break

                    # if cap is full, check next object
                    if remaining_caps[obj] == 0:
                        next_proposing_ranks[a] += 1
                        continue

                    # otherwise add an edge from a to obj
                    edges.append([f"a_{a}", f"o_{obj}"])
                    break

            # add edges from objects to agents
            edges_from_obj = {obj: [] for obj in range(self.num_objects)}
            for a, obj in enumerate(current_init_assignment):
                if obj != self.outside_option:
                    edges_from_obj[obj].append(a)

            for obj, li in edges_from_obj.items():
                for a in li:
                    edges.append([f"o_{obj}", f"a_{a}"])

                if remaining_caps[obj] - len(li) > 0:
                    edges.append([f"o_{obj}", f"a_{priority_a}"])

            # find cycle
            graph = nx.DiGraph()
            graph.add_edges_from(edges)
            cycle = nx.find_cycle(graph)
            
             # update matching using cycle
            for edge in cycle:
                if edge[0].startswith("a_"):
                    a = int(edge[0].split("_")[1])
                    obj = int(edge[1].split("_")[1])
                    matching[a] = obj
                    current_init_assignment[a] = self.outside_option
                    if obj != self.outside_option:
                        remaining_caps[obj] -= 1

        return matching


    def top_trading_cycles(self, initial_assignment):
        """
        Run the top trading cycles algorithm in a one-sided matching market.
        This method is an instance of YRMH_IGYT algorithm.

        Args:
            initial_assignment : 1d-array(int)
                List of the initial assignments of the agents.

        Returns:
            matching : 1d-ndarray
                List of the matched objects. The n-th element indicates 
                the object that the n-th agent matches.
        """

        return self.YRMH_IGYT(initial_assignment)


    def get_agent_matching_ranks(self, matching, rank_table=None):
        if rank_table is None:
            rank_table = self.convert_prefs_to_ranks(
                self.agent_prefs, 
                self.num_objects
            )
        
        return self.get_matching_ranks(matching, rank_table)


if __name__ == "__main__":
    """
    d_prefs = [
        [0, 2, 1, 3], 
        [1, 0, 2, 3], 
        [0, 1, 2, 3], 
        [2, 0, 1, 3], 
    ]
    h_prefs = [
        [0, 2, 1, 3, 4], 
        [1, 0, 2, 3, 4], 
        [2, 0, 3, 1, 4], 
    ]
    caps = np.array([1, 1, 1])
    m = ManyToOneMarket(d_prefs, h_prefs, caps)
    r1 = m.deferred_acceptance()
    r2 = m.top_trading_cycles()
    print(r1)
    print(r2)

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
    print(m.top_trading_cycles())
    print(m.serial_dictatorship())
    """
    pass
