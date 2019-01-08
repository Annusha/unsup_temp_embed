#!/usr/bin/python2.7

import numpy as np
import cv2
import skvideo.io
import argparse


# global definition of colors (supports up to len(colors) many classes per video)
colors = [ (255, 0, 0),
           (0, 255, 0),
           (0, 0, 255),
           (255, 255, 0),
           (255, 0, 255),
           (0, 255, 255),
           (255, 125, 125),
           (125, 255, 125),
           (125, 125, 255),
           (255, 255, 125),
           (255, 125, 255),
           (25, 125, 255),
           (255, 125, 25),
           (255, 12, 25),
           (0, 125, 255),
           (255, 125, 0),
           (11, 50, 200),
           (25, 125, 55),
           (125, 255, 255) ]

np.random.shuffle(colors)


# offset_y for text
offset_y = 100


class Segmentation(object):
    def __init__(self, image, labels):
        self.image = image
        self.labels = labels


def read_segmentation(filename):
    with open(filename, 'r') as f:
        content = f.read().split('\n')[0:-1]
    return content


def generate_segmentation(segmentation, color_map, width):
    image = np.zeros((20, width, 3), dtype=np.uint8)
    norm = len(segmentation) / float(width)
    segment_labels = [0] * width
    # fill image with values
    for t in range(width):
        segment_labels[t] = segmentation[ int(t * norm) ]
        image[:, t, 0] = color_map[ segment_labels[t] ][0]
        image[:, t, 1] = color_map[ segment_labels[t] ][1]
        image[:, t, 2] = color_map[ segment_labels[t] ][2]
    return Segmentation(image, segment_labels)


def add_segmentation(frame, t, n_frames, segmentation, offset_x):
    offset_x = frame.shape[0] - offset_x
    # add segmentation
    frame[offset_x - segmentation.image.shape[0] : offset_x, offset_y : offset_y + segmentation.image.shape[1], :] = segmentation.image
    # add position indicator
    pos = offset_y + int( float(t) / n_frames * segmentation.image.shape[1] )
    frame[offset_x - segmentation.image.shape[0] : offset_x, pos-1 : pos+2, :] = np.zeros((segmentation.image.shape[0], 3, 3), dtype=np.uint8)


def render_frame(v_out, frame, t, n_frames, prediction, ground_truth):
    #frame[frame.shape[0] - 105 : frame.shape[0], :, :] = 0
    # add segmentations
    offset_x = 5
    add_segmentation(frame, t, n_frames, ground_truth, offset_x)
    offset_x = 10 + ground_truth.image.shape[0]
    add_segmentation(frame, t, n_frames, prediction, offset_x)
    # put text
    cv2.putText(frame, 'prediction', (5, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'gr. truth', (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    # put label text
    #frame[frame.shape[0] - 140 : frame.shape[0] - 80, 0:270, :] = 0
    cv2.putText(frame, 'prediction:', (5, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'correct:', (5, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    pos_pred = int( float(t) / n_frames * len(prediction.labels) )
    pos_gt = int( float(t) / n_frames * len(ground_truth.labels) )
    color = (50, 205, 50)
   
    if ground_truth.labels[pos_gt] != prediction.labels[pos_pred]:
        color = (255, 0, 0)
    cv2.putText(frame, prediction.labels[pos_pred], (offset_y+5, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
    cv2.putText(frame, ground_truth.labels[pos_gt], (offset_y+5, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, .6, (50, 205, 50), 1, cv2.LINE_AA)
    # put timer on frame
    cv2.putText(frame, '(2x speed)', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    # write frame
    v_out.writeFrame(frame)


def render(prediction, ground_truth, width, height, v_in, v_out):
    # generate color map
    color_map = dict()
    label_set = set( prediction + ground_truth )
    for i, label in enumerate(label_set):
        color_map[label] = colors[i]
    # generate segmentations
    prediction = generate_segmentation(prediction, color_map, width - offset_y - 10)
    ground_truth = generate_segmentation(ground_truth, color_map, width - offset_y - 10)
    # write video
    n_frames = int(v_in.get(cv2.CAP_PROP_FRAME_COUNT))
    for t in range(0, n_frames, 1): # ATTENTION: HARD CODED SPEEDUP OF 2
        ret, frame = v_in.read()
	# if t%4:
     #        continue
        if ret == True:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            render_frame(v_out, cv2.resize(frame, (width, height)), t, n_frames, prediction, ground_truth)
        else:
            render_frame(v_out, np.zeros((height, width, 3), dtype=np.uint8), t, n_frames, prediction, ground_truth)



def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video', type=str)
    parser.add_argument('prediction', type=str)
    parser.add_argument('ground_truth', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--title', type=str, default='Example video from the Breakfast dataset')
    args = parser.parse_args()

    v_in = cv2.VideoCapture(args.input_video)
    v_out = skvideo.io.FFmpegWriter(args.output_file,
                                    inputdict={'-r': str(args.fps)},
                                    outputdict={'-c:v': 'mpeg4', '-b': '50150320', '-r': str(args.fps), '-vf': 'scale=' + str(args.width) + ':' + str(args.height)}
                                   )
    # print title frame
    textsize = cv2.getTextSize(args.title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    #cv2.putText(frame, args.title, (args.width / 2 - textsize[0] / 2, args.height / 2 + textsize[1] / 2), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.CV_AA)
    cv2.putText(frame, args.title, (args.width // 2 - textsize[0] // 2 + 80, args.height // 2 + textsize[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)
    for t in range(3 * int(args.fps)):
        v_out.writeFrame(frame)
    # render remaining video
    render( read_segmentation(args.prediction), read_segmentation(args.ground_truth), args.width, args.height, v_in, v_out )
    v_in.release()
    v_out.close()


if __name__ == '__main__':
    main()
