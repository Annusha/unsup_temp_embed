from os.path import join
import numpy as np

from utils.arg_pars import opt


class Grammar(object):
    def __init__(self, states):
        self._states = states
        self._grammar = [0]
        self._framewise_states = []
        self._state2label = {}

    def _init_positions(self):
        accumulator = 0
        for s in self._states:
            self._positions.append(accumulator)
            accumulator += s

        for g in self._grammar:
            start = self._positions[g]
            self._grammar_states += range(start, start + self._states[g])

    def framewise_states(self, last=False):
        if last:
            # to return result likelihood
            return self._framewise_states[0]
        frame_wise_labels = []
        for state in reversed(self._framewise_states):
            frame_wise_labels.append(self._state2label[self._grammar_states[state]])
        return frame_wise_labels

    def add_framewise_state(self, states, last=False):
        if last:
            if not isinstance(states, np.ndarray):
                # define last state of the grammar strictly
                state = int(states)
            else:
                start = len(states) - self._states[self._grammar[-1]]
                state = np.argmin(states[start:]) + start
        else:
                state = int(states[self._framewise_states[-1]])
        self._framewise_states.append(state)

    def __len__(self):
        return len(self._grammar_states)

    def states(self):
        return self._grammar_states
