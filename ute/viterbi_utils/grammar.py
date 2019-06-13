#!/usr/bin/env python

"""Grammar implementation for the viterbi decoding. Object should be assign to
each video to collect the states for each of the frames.
"""

__author__ = 'Anna Kukleva'
__date__ = 'September 2018'


import numpy as np


class Grammar:
    def __init__(self, states):
        """
        Args:
            states: flat sequence (list) of states (class State)
        """
        self._states = states
        self._framewise_states = []
        self.name = '%d' % len(states)

    def framewise_states(self):
        return_states = list(map(lambda x: self._states[x], self._framewise_states))
        return return_states

    def reverse(self):
        self._framewise_states = list(reversed(self._framewise_states))

    def __getitem__(self, idx):
        return self._framewise_states[idx]

    def set_framewise_state(self, states, last=False):
        """Set states for each item in a sequence.
        Backward pass by setting a particular state for computed probabilities.
        Args:
            states: either state indexes of the previous step
            last: if it the last item or not

        Returns:

        """
        if not last:
            state = int(states[[self._framewise_states[-1]]])
        else:
            # state = int(self._states[-1])
            state = int(len(self._states) - 1)

        self._framewise_states.append(state)

    def states(self):
        return self._states

    def __len__(self):
        return len(self._states)


def create_grammar(n_states):
    """Create grammar out of given number of possible states with 1 sub-action
    each.
    """
    states = list(range(n_states))
    grammar = Grammar(states)
    return grammar
