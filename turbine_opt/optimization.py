import math
import numpy as np

from turbine_opt.lib import cpt_blades_imbalance
from turbine_opt.lib import naive_blade_swap
from turbine_opt.lib import boosted_blade_swap


def random_neighbour_choice(disk, energy):
    """
    swap a blade from group 1 with a blade from group 2
    if energy > 40: use boosted blade swap
    if energy < 40: use naive blade swap
    """
    if energy > 40:
        return boosted_blade_swap(disk.copy())
    else:
        return naive_blade_swap(disk.copy())


def cost_function_unbalance(disk):
    """
    Cost of disk
    :input:
        disk :: pd.DataFrame read from read_data
    """
    fx, fy = cpt_blades_imbalance(disk)
    unbalance_blades = math.sqrt(fx.sum() * fx.sum() + fy.sum() * fy.sum())
    return unbalance_blades


def temperature(k, kmax):
    """
    decreasing temperature function
    :input:
        k :: current iteration
        kmax :: max number of iterations
    """
    return max(0.01, min(1, 1 - float(k)/float(kmax)))


def acceptance_probability(energy, new_energy, temperature):
    """
    acceptance probability function, which decreases with
    temperature
    :input:
        energy :: energy value from current state
        new_energy :: energy value from new state
        temperature :: value of temperature from temperature function
    """
    if new_energy < energy:
        return 1
    else:
        p = np.exp(- (new_energy - energy) / temperature)
        return p


class Simulated_annealing:
    """
    encapsulated simulated annealing class
    """
    def __init__(self, init_state, chg_state_func, ener_func, temp_func,
                 max_steps, accept_proba):
        """
        constructor
        :input:
            init_state :: initial state, e.g. initial vector
            chg_state_func :: function to change update state at each iteration
            ener_func :: energy function
            temp_func :: temperature function
            max_steps :: max number of steps
            accept_proba :: probabilistic function of acceptance of new states
        """
        self.init_state = init_state
        self.chg_state_func = chg_state_func
        self.ener_func = ener_func
        self.temp_func = temp_func
        self.max_steps = max_steps
        self.accept_proba = accept_proba

    def optimize(self):
        """
        optimization method
        """
        state = self.init_state.copy()
        final_state = self.init_state.copy()
        state_new = self.init_state.copy()
        for k in range(self.max_steps):
            final_en = self.ener_func(final_state)
            temp = self.temp_func(k, self.max_steps)
            state_new.loc[:, ['w',
                              'blade',
                              'ht']] = self.chg_state_func(state,
                                                           final_en)[['w',
                                                                      'blade',
                                                                      'ht']]
            state_ener = self.ener_func(state)
            state_nw_ener = self.ener_func(state_new)
            proba = self.accept_proba(state_ener,
                                      state_nw_ener,
                                      temp)
            # print(final_en, ": self.ener_func(final state)")
            if proba > np.random.sample():
                state = state_new.copy()
            if self.ener_func(state) < final_en:
                final_state = state.copy()
        return final_state
