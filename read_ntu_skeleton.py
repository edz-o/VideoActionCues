import json
import os
import os.path as osp
import numpy as np

class Joint(object):
    def __init__(self, jointinfo):
        self.x = float(jointinfo[0]);
        self.y = float(jointinfo[1]);
        self.z = float(jointinfo[2]);

        self.depthX = float(jointinfo[3]);
        self.depthY = float(jointinfo[4]);

        self.colorX = float(jointinfo[5]);
        self.colorY = float(jointinfo[6]);

        self.orientationW = float(jointinfo[7]);
        self.orientationX = float(jointinfo[8]);
        self.orientationY = float(jointinfo[9]);
        self.orientationZ = float(jointinfo[10]);

class Pose(object):
    def __init__(self, info):
        self.info = info

    def get_bbox(self, x_scale, y_scale):
        bbox = [99999, 99999, -1, -1]
        for joint in self.info['joints']:
            bbox[0] = min(joint.colorX*x_scale/1920, bbox[0])
            bbox[1] = min(joint.colorY*y_scale/1080, bbox[1])
            bbox[2] = max(joint.colorX*x_scale/1920, bbox[2])
            bbox[3] = max(joint.colorY*y_scale/1080, bbox[3])
        return [int(x) for x in bbox]

def read_ntu_skeleton_file(filename):
    body_info = []
    with open(filename, 'r') as f:
        frames = int(f.readline().strip())

        for frame in range(frames):
            info = []
            bodycount = int(f.readline().strip())
            for b in range(bodycount):
                info_b = f.readline().strip().split(' ')
                body = dict(
                            bodyID=int(info_b[0]),
                            clipedEdges=int(info_b[1]),
                            handLeftConfidence=int(info_b[2]),
                            handLeftState=int(info_b[3]),
                            handRightConfidence=int(info_b[4]),
                            handRightState=int(info_b[5]),
                            isResticted=int(info_b[6]),
                            leanX=float(info_b[7]),
                            leanY=float(info_b[8]),
                            trackingState=int(info_b[9]),
                           )
                jointcount = int(f.readline().strip())
                joints = []
                for j in range(jointcount):
                    jointinfo = f.readline().strip().split(' ')
                    joint = Joint(jointinfo)
                    joints.append(joint)
                body.update(dict(joints=joints))
                info.append(body)
            body_info.append(info)
    return body_info

def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)

"""
skeleton = read_ntu_skeleton_file('data/nturgbd/skeletons_val/S001C001P001R001A007.skeleton')
img_copy = img.copy()
pose = Pose(skeleton[frame][body_id])
for joint in pose.info['joints']:
    import pdb
    pdb.set_trace()
    #cv2.circle(img_copy, (int(joint.colorX/1920*640), int(joint.colorY/1080*360)), radius=2, color=(0,255,0), thickness=-1)

"""