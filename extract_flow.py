import torch
import cv2
import numpy as np
import os
import sys


def extract_flow(frame_list, frame_names, output_dir):
    frame1 = frame_list[0]
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    for frame, frame_name in zip(frame_list[1:], frame_names[1:]):
        nxt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        frame_name = os.path.splitext(frame_name)[0]
        cv2.imwrite(os.path.join(output_dir, frame_name + '_opticalhsv.png'), rgb)
        prvs = nxt


def main():
    if len(sys.argv) < 3:
        print('Usage: python extract_flow.py <INPUT FOLDER> <OUTPUT FOLDER>')
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    for (dirpath, video_names, image_names) in os.walk(input_dir):
        if '_rgb' not in dirpath and len(image_names) < 1:
            continue

        video_name = os.path.split(dirpath)[-1]
        video_output_dir = os.path.join(output_dir, video_name)
        try: 
            os.mkdir(video_output_dir)
        except OSError:
            continue

        frame_list = [cv2.imread(os.path.join(dirpath, image_name)) for
                      image_name in sorted(image_names)]
        extract_flow(frame_list, sorted(image_names), video_output_dir)
        #print('extracted flow for {}'.format(video_name))


if __name__ == '__main__':
    main()

