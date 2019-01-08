#!/usr/bin/python2.7

import numpy as np
from viterbi_utils.src.utils.grammar import PathGrammar
from viterbi_utils.src.utils.length_model import PoissonModel
import glob
import re

# Viterbi decoding
class Viterbi(object):

    ### helper structure ###
    class TracebackNode(object):
        def __init__(self, label, predecessor, boundary = False):
            self.label = label
            self.predecessor = predecessor
            self.boundary = boundary

    ### helper structure ###
    class HypDict(dict):
        class Hypothesis(object):
            def __init__(self, score, traceback):
                self.score = score
                self.traceback = traceback

        def update(self, key, score, traceback):
            if (not key in self) or (self[key].score <= score):
                self[key] = self.Hypothesis(score, traceback)

    # @grammar: the grammar to use, must inherit from class Grammar
    # @length_model: the length model to use, must inherit from class LengthModel
    # @frame_sampling: generate hypotheses every frame_sampling frames
    # @max_hypotheses: maximal number of hypotheses. Smaller values result in stronger pruning
    def __init__(self, grammar, length_model, frame_sampling = 1, max_hypotheses = np.inf):
        self.grammar = grammar
        self.length_model = length_model
        self.frame_sampling = frame_sampling
        self.max_hypotheses = max_hypotheses

    # Viterbi decoding of a sequence
    # @log_frame_probs: logarithmized frame probabilities
    #                   (usually log(network_output) - log(prior) - max_val, where max_val ensures negativity of all log scores)
    # @return: the score of the best sequence,
    #          the corresponding framewise labels (len(labels) = len(sequence))
    #          and the inferred segments in the form (label, length)
    def decode(self, log_frame_probs):
        assert log_frame_probs.shape[1] == self.grammar.n_classes()
        frame_scores = np.cumsum(log_frame_probs, axis=0) # cumulative frame scores allow for quick lookup if frame_sampling > 1
        print('Found ' + str(frame_scores.shape[0]) + ' frames')
        # create initial hypotheses
        hyps = self.init_decoding(frame_scores)
        # decode each following time step
        #for t in range(2 * self.frame_sampling - 1, frame_scores.shape[0], self.frame_sampling):
        for t in range(0, frame_scores.shape[0], self.frame_sampling):
            # if t % 500 == 0:
            #     print('Processing frame ' + str(t));
            hyps = self.decode_frame(t, hyps, frame_scores)
            self.prune(hyps)
            #print(hyps)
        # transition to end symbol
        final_hyp = self.finalize_decoding(hyps)
        labels, segments = self.traceback(final_hyp, frame_scores.shape[0])
        # print(labels)
        # print(len(labels))
        # print(frame_scores.shape[0])
        # print(segments)
        labels = labels[:frame_scores.shape[0]];
        # print(len(labels))
        return final_hyp.score, labels, segments


    ### helper functions ###
    def frame_score(self, frame_scores, t, label):
        if t >= self.frame_sampling:
            return frame_scores[t, label] - frame_scores[t - self.frame_sampling, label]
        else:
            return frame_scores[t, label]

    def prune(self, hyps):
        if len(hyps) > self.max_hypotheses:
            tmp = sorted( [ (hyps[key].score, key) for key in hyps ] )
            del_keys = [ x[1] for x in tmp[0 : -self.max_hypotheses] ]
            for key in del_keys:
                del hyps[key]

    def init_decoding(self, frame_scores):
        hyps = self.HypDict()
        context = (self.grammar.start_symbol(),)
        for label in self.grammar.possible_successors(context):
            key = context + (label, self.frame_sampling)
            score = self.grammar.score(context, label) + self.frame_score(frame_scores, self.frame_sampling - 1, label)
            hyps.update(key, score, self.TracebackNode(label, None, boundary = True))
        return hyps

    def decode_frame(self, t, old_hyp, frame_scores):
        new_hyp = self.HypDict()
        for key, hyp in old_hyp.items():
            context, label, length = key[0:-2], key[-2], key[-1]
            # stay in the same label...
            new_key = context + (label, min(length + self.frame_sampling, self.length_model.max_length()))
            score = hyp.score + self.frame_score(frame_scores, t, label)
            new_hyp.update(new_key, score, self.TracebackNode(label, hyp.traceback, boundary = False))
            # ... or go to the next label
            context = context + (label,)
            for new_label in self.grammar.possible_successors(context):
                if new_label == self.grammar.end_symbol():
                    continue
                new_key = context + (new_label, self.frame_sampling)
                score = hyp.score + self.frame_score(frame_scores, t, label) + self.length_model.score(length, label) + self.grammar.score(context, new_label)
                new_hyp.update(new_key, score, self.TracebackNode(new_label, hyp.traceback, boundary = True))
        # return new hypotheses
        return new_hyp

    def finalize_decoding(self, old_hyp):
        final_hyp = self.HypDict.Hypothesis(-np.inf, None)
        for key, hyp in old_hyp.items():
            context, label, length = key[0:-2], key[-2], key[-1]
            context = context + (label,)
            score = hyp.score + self.length_model.score(length, label) + self.grammar.score(context, self.grammar.end_symbol())
            if score >= final_hyp.score:
                final_hyp.score, final_hyp.traceback = score, hyp.traceback
        # return final hypothesis
        return final_hyp

    def traceback(self, hyp, n_frames):
        class Segment(object):
            def __init__(self, label):
                self.label, self.length = label, 0
        traceback = hyp.traceback
        labels = []
        segments = [Segment(traceback.label)]
        while not traceback == None:
            segments[-1].length += self.frame_sampling
            labels += [traceback.label] * self.frame_sampling
            if traceback.boundary and not traceback.predecessor == None:
                segments.append( Segment(traceback.predecessor.label) )
            traceback = traceback.predecessor
        segments[0].length += n_frames - len(labels) # append length of missing frames
        labels += [hyp.traceback.label] * (n_frames - len(labels)) # append labels for missing frames
        return list(reversed(labels)), list(reversed(segments))


