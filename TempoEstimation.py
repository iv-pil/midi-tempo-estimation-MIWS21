# File: TempoEstimation.py
# Author: Ivan Pilkov
# Last edited: 2022-01-24

import partitura as pt
import numpy as np
import itertools


class Cluster(object):
    """
    Similarity cluster of IOIs
    """
    def __init__(self, ioi):
        #self.id = pass #do we need an actual id? or just use the interval
        self.iois = [ioi]
        self.interval = 0
        self.get_interval()
        self.score = 0

        #del flag to remove unused cluster
        self.delete = False

    def __len__(self):
        return len(self.iois)

    def get_interval(self):
        self.interval = np.mean(self.iois)

    def get_score(self):
        self.score = self.__len__()

    def add(self, ioi):
        self.iois.append(ioi)

    def assimilate(self, other_cluster):
        '''for new_ioi in other_cluster.iois:
            self.iois.append(new_ioi)'''
        self.iois += other_cluster.iois


class RhythmicEvent(object):
    def __init__(self, onset):
        self.onset = onset
        self.notes = []
        #self.saliency = 0

    def add_note(self, note):
        self.notes.append(note)

    def set_saliency(self, note):
        pass


class Agent(object):
    """
    Beat-tracking Agent
    """
    def __init__(self, tempo, event):
        self.beat_interval = tempo
        self.history = [event]
        self.score = event['saliency']
        self.prediction = 0
        self.get_prediction(event['onset'])
        self.tw = 0.05 * tempo

        # delete flag
        self.delete = False

    def get_prediction(self, beat_time):
        self.prediction = self.beat_interval + beat_time

    def update_score(self, event, error):
        self.score += (1-error/(2*self.tw))*event['saliency']


# helper functions
def trunc(pitch):
    if pitch < 48:
        return 48
    if pitch > 72:
        return 72
    return pitch


def saliency(event):
    d = max([note['note_off'] - note['note_on'] for note in event['notes']])
    p = min([trunc(note['midi_pitch']) for note in event['notes']])
    v = sum([note['velocity'] for note in event['notes']])
    # for all-positive saliency maybe use bellow
    #return 300*d - 4*p + v
    #return 300*d + (127-p) + v
    # non-linear variant
    return min(1000, d * (84-p) * np.log(v))


def main(path_to_midi):

    ppart = pt.load_performance_midi(path_to_midi)

    # list of dicts containing all rhythmic events
    events = [{'onset': 0,
               'notes': []
               }]

    # using 75ms as a starting point
    SPREAD_TIME = 0.075

    #TODO: make events a list of objects
    for note in ppart.notes:
        previous_onset = events[-1]['onset']
        current_onset = note['note_on']
        if abs(current_onset - previous_onset) < SPREAD_TIME:
            events[-1]['notes'].append(note)
        else:
            events.append({'onset': current_onset,
                           'notes': [note]})

    # calculate saliency for each event
    # add to dictionary and discard all other note information
    for event in events:
        event['saliency'] = saliency(event)
        del event['notes']

    CLUSTER_WIDTH = 0.025
    clusters = []

    # create and populate cluster with IOIs --change it to RhythtmicEvent objects
    for ei, ej in list(itertools.combinations(events, 2)):
        ioi = abs(ei['onset']-ej['onset'])

        # fixed - now list is populated by clusters
        candidates = [c for c in clusters if abs(c.interval - ioi) < CLUSTER_WIDTH]

        if len(candidates):
            winner = min(candidates, key=lambda x: x.interval)
            winner.add(ioi)
            winner.get_interval()
        else:
            clusters.append(Cluster(ioi))

    for ci, cj in list(itertools.combinations(clusters, 2)):
        if abs(ci.interval-cj.interval) < CLUSTER_WIDTH:
            ci.assimilate(cj)
            ci.get_interval()
            cj.delete = True
            #debugging
            #print(len(ci))

    # go through list again, score all clusters and remove all flagged cluster
    for c in clusters[:]:
        c.get_score()
        if c.delete:
            clusters.remove(c)

    # iterate through all pairs of clusters left and assign each one a score
    for ci, cj in list(itertools.combinations(clusters, 2)):
        for n in range(2, 9):
            if abs(ci.interval - n*cj.interval) < CLUSTER_WIDTH:
                #ci.score += (6-n if n<4 else 1)*len(cj) # orig
                cj.score += (6-n if n<4 else 1)*len(ci)

    clusters.sort(key=lambda x: x.score, reverse=True)

    # setting an upper limit on possible inter-beat intervals (pos_tempi)
    # based on clusters' score
    NBEST_CLUSTERS = 50
    #choose StartupPeriod in s, in which initial tracking begins
    STARTUP = 3
    # max len of time for unsuccessful tracking
    TIMEOUT = 3

    # inner window taking care of beat mismatch
    T_INNER = 0.05 #0.04

    #Error corr factor - inversely proportional to beat interval update
    CORRFACTOR = 2#1.5

    #Threshold for comparison of agents' similarity in prediction
    THRESHOLD = 0.02

    pos_tempi = [c.interval for c in clusters[:NBEST_CLUSTERS]]

    agents = []
    for tempo in pos_tempi:
        for event in events:
            if event['onset'] < STARTUP:
                agents.append(Agent(tempo, event))
    new_agents = []
    for event in events:
        onset = event['onset']
        for agent in agents:
            if onset - agent.history[-1]['onset'] > TIMEOUT:
                #print(onset, agent.history[-1]['onset'])
                agent.delete = True

                #print('break')
                break

            while agent.prediction + agent.tw < onset:
                agent.get_prediction(agent.prediction)
            if agent.prediction - agent.tw <= onset <= agent.prediction + agent.tw:
                if abs(agent.prediction - onset) > T_INNER:
                    new_agents.append(Agent(agent.beat_interval, event))

                    #i += 1

                error = onset - agent.prediction
                agent.beat_interval += error/CORRFACTOR
                agent.get_prediction(onset)
                agent.history.append(event)
                agent.update_score(event, error)
                #print(i)

        for agent in agents[:]:
            if agent.delete:
                agents.remove(agent)
                #j += 1
        #print('first', len(agents))

        #TODO: delete duplicate agents
        for ai, aj in itertools.combinations(agents, 2):
            if abs(ai.beat_interval - aj.beat_interval) < THRESHOLD and abs(ai.prediction - aj.prediction) < THRESHOLD:
                if ai.score > aj.score:
                    aj.delete = True
                else:
                    ai.delete = True

        for agent in agents[:]:
            if agent.delete:
                agents.remove(agent)
                #k += 1
        #print('second', len(agents))


    agents += new_agents

    agents.sort(key=lambda x: x.score, reverse=True)
    beat_interval = agents[0].beat_interval
    predicted_tempo = 60*(1/beat_interval)

    return predicted_tempo


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Test Tempo Estimation')
    parser.add_argument('--MIDIfile', '-i',
                        help='path to MIDI input file',
                        default="",
                        type=str)
    parser.add_argument('--outdir', '-o',
                        help='Output text file directory',
                        type=str,
                        default=".")
    args = parser.parse_args()

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    midi_name = os.path.splitext(os.path.basename(args.MIDIfile))[0]

    tempo_in_bpms_pred = main(args.MIDIfile)

    print(f"Predicted tempo in bpm: {tempo_in_bpms_pred}")
    print("Writing to file...")

    outfile = os.path.join(
        args.outdir,
        f"{script_name}_{midi_name}_results.txt"
    )

    with open(outfile, "w") as f:
        f.write(str(tempo_in_bpms_pred))
    print("Finished.")
