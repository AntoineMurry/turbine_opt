import numpy as np


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
