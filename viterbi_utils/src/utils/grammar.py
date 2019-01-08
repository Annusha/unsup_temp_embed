#!/usr/bin/python2.7

import numpy as np


class Grammar(object):
    
    # @context: tuple containing the previous label indices
    # @label: the current label index
    # @return: the log probability of label given context p(label|context)
    def score(self, context, label): # score is a log probability
        return 0.0

    # @return: the number of classes
    def n_classes(self):
        return 0

    # @return sequence start symbol
    def start_symbol(self):
        return -1

    # @return sequence end symbol
    def end_symbol(self):
        return -2

    # @context: tuple containing the previous label indices
    # @return: list of all possible successor labels for the given context
    def possible_successors(context):
        return set()


class PathGrammar(Grammar):
    
    def __init__(self, transcript_file, label2index_map):
        self.num_classes = len(label2index_map)
        transcripts = self._read_transcripts(transcript_file, label2index_map)
        # generate successor sets
        self.successors = dict()
        # if len(transcripts) < 2 :
        #     # very bad hack!
        #     transcripts.append(transcripts[0]);
        #
        for transcript in transcripts:
            transcript = transcript + [self.end_symbol()]
            # transcript = transcript.append(self.end_symbol());
            for i in range(len(transcript)):
                context = (self.start_symbol(),) + tuple(transcript[0:i])
                print(context)
                self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )
                print(self.successors[context])

    def _read_transcripts(self, transcript_file, label2index_map):
        transcripts = []
        with open(transcript_file, 'r') as f:
            lines = f.read().split('\n')[0:-1]
        for line in lines:
            transcripts.append( [ label2index_map[label.strip()] for label in line.split() ] )
            #for label in line.split():
                #transcripts.append( label2index_map[label.strip()]  )
        print(str(len(transcripts))+" transcripts found ...")
        return transcripts

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf


class SingleTranscriptGrammar(Grammar):

    def __init__(self, transcript, n_classes):
        """
        Args:
            transcript: list of action labels
            n_classes: number of classes
        """
        self.num_classes = n_classes
        transcript = transcript + [self.end_symbol()]
        self.successors = dict()
        for i in range(len(transcript)):
            context = (self.start_symbol(),) + tuple(transcript[0:i])
            self.successors[context] = set([transcript[i]]).union( self.successors.get(context, set()) )

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf



