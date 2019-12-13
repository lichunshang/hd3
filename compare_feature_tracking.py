from __future__ import absolute_import, division, print_function
from copy import deepcopy
import skimage.io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.blocking_input import BlockingInput
import natsort
import hd3model as models
import torch
import math
from PIL import Image
import torch.nn.functional as F
import data.flowtransforms as transforms
import utils.flowlib  as fl
from models import hd3_ops


def get_target_size(H, W):
    h = 64 * np.array([[math.floor(H / 64), math.floor(H / 64) + 1]])
    w = 64 * np.array([[math.floor(W / 64), math.floor(W / 64) + 1]])
    ratio = np.abs(np.matmul(np.transpose(h), 1 / w) - H / W)
    index = np.argmin(ratio)
    return h[0, index // 2], w[0, index % 2]


def resize_dense_vector(vec, des_height, des_width):
    ratio_height = float(des_height / vec.size(2))
    ratio_width = float(des_width / vec.size(3))
    vec = F.interpolate(
            vec, (des_height, des_width), mode='bilinear', align_corners=True)
    if vec.size(1) == 1:
        vec = vec * ratio_width
    else:
        vec = torch.stack(
                [vec[:, 0, :, :] * ratio_width, vec[:, 1, :, :] * ratio_height],
                dim=1)
    return vec


class BlockingKeyInput(BlockingInput):
    """
    Callable for retrieving mouse clicks and key presses in a blocking way.
    """

    def __init__(self, fig):
        # BlockingInput.__init__(self, fig=fig, eventslist=(
        #     'button_press_event', 'key_press_event'))
        BlockingInput.__init__(self, fig=fig, eventslist=('key_press_event',))
        self.event_key = None

    def post_event(self):
        """Determine if it is a key event."""
        if self.events:
            self.keyormouse = self.events[-1].name == 'key_press_event'
            self.event_key = self.events[-1].key

    def __call__(self, timeout=30):
        """
        Blocking call to retrieve a single mouse click or key press.

        Returns ``True`` if key press, ``False`` if mouse click, or ``None`` if
        timed out.
        """
        self.keyormouse = None
        BlockingInput.__call__(self, n=1, timeout=timeout)

        return self.event_key


class Compare(object):
    def __init__(self):
        # Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
        ckpt_path = '/home/cs4li/Dev/hd3/model_zoo/hd3fc_chairs_things_kitti-bfa97911.pth'
        self.corr_range = [4, 4, 4, 4, 4, 4]
        self.corr_range = self.corr_range[:5]
        self.nn = models.HD3Model("flow", "dlaup", "hda", self.corr_range, True).cuda()

        self.nn = torch.nn.DataParallel(self.nn).cuda()

        checkpoint = torch.load(ckpt_path)
        self.nn.load_state_dict(checkpoint['state_dict'], strict=True)
        # self.nn = self.nn.module
        self.nn.eval()

        # transform
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.torch_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)])

    def compute_nn_flow(self, im1, im2, cur_pts):
        im1 = np.stack([im1, im1, im1], axis=-1)
        im2 = np.stack([im2, im2, im2], axis=-1)
        ret = self.nn.predict_from_img_pairs(((im1, im2,),), batch_size=1, verbose=False)
        flow_sampled = ret[0][cur_pts.astype(np.uint)[:, 1], cur_pts.astype(np.uint)[:, 0], :]

        return cur_pts + flow_sampled, ret[0]

    def compute_flow_hd3(self, im1, im2, cur_pts):
        th, tw = get_target_size(im1.height, im1.width)
        im_tensor = self.torch_transform([im1, im2], [])[0]
        im1_tensor = F.interpolate(im_tensor[0].unsqueeze(0), (int(round(th)), int(round(tw))), mode='bilinear',
                                   align_corners=True)
        im2_tensor = F.interpolate(im_tensor[1].unsqueeze(0), (int(round(th)), int(round(tw))), mode='bilinear',
                                   align_corners=True)

        output = self.nn(
                img_list=[im1_tensor, im2_tensor],
                label_list=[],
                get_vect=True,
                get_prob=True,
                get_epe=False)
        scale_factor = 1 / 2 ** (7 - len(self.corr_range))
        output['vect'] = resize_dense_vector(output['vect'] * scale_factor,
                                             im1.height,
                                             im1.width)

        flow = output['vect'].detach().cpu().numpy()[0]
        flow = np.stack([flow[0], flow[1]], axis=-1)
        flow_sampled = flow[cur_pts.astype(np.uint)[:, 1], cur_pts.astype(np.uint)[:, 0], :]

        # uncertainty = resize_dense_vector(output['prob'] * scale_factor, im1.height, im1.width)
        # uncertainty = uncertainty.detach().cpu().numpy()[0]

        confidence_map = hd3_ops.prob_gather(output['prob'].detach().cpu(), dim=2)
        confidence_map = F.interpolate(confidence_map, (im1.height, im1.width), mode='nearest')

        return cur_pts + flow_sampled, flow, confidence_map.squeeze()

    def compare_optical_flow(self, plt_name, im1, im2, fig, ax):
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        im1_cv = skimage.img_as_ubyte(im1)
        im2_cv = skimage.img_as_ubyte(im2)
        im1_cv = clahe.apply(im1_cv)
        im2_cv = clahe.apply(im2_cv)
        cur_pts = cv2.goodFeaturesToTrack(im1_cv, 150, 0.01, 25)
        nxt_pts_klt, status, err = cv2.calcOpticalFlowPyrLK(im1_cv, im2_cv, cur_pts, None, winSize=(21, 21,),
                                                            maxLevel=3)

        cur_pts = cur_pts.squeeze()
        # nxt_pts_nn, flow = self.compute_nn_flow(im1_cv, im2_cv, cur_pts)
        nxt_pts_nn, flow, confidence = self.compute_flow_hd3(Image.fromarray(im1_cv).convert("RGB"),
                                                             Image.fromarray(im2_cv).convert("RGB"), cur_pts)
        nxt_pts_lkr, status_lkr, err_lkr = cv2.calcOpticalFlowPyrLK(im1_cv, im2_cv, cur_pts, nxt_pts_nn.copy(),
                                                                    winSize=(21, 21,),
                                                                    maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

        nxt_pts_klt = nxt_pts_klt.squeeze()
        nxt_pts_klt = nxt_pts_klt[status.squeeze() == 1]
        cur_pts_filtered = cur_pts[status.squeeze() == 1]

        confidence_scaled = confidence * 255
        ax[0, 0].clear()
        ax[0, 0].imshow(im1_cv, cmap='gray')
        ax[0, 0].scatter(cur_pts[:, 0], cur_pts[:, 1], s=3, c="b")

        ax[0, 2].clear()
        ax[0, 2].imshow(confidence_scaled, cmap='gray')
        ax[0, 2].scatter(cur_pts[:, 0], cur_pts[:, 1], s=3, c="b")

        ax[0, 1].clear()
        ax[0, 1].imshow(im2_cv, cmap='gray')
        ax[0, 1].scatter(cur_pts[:, 0], cur_pts[:, 1], s=3, c="b")
        ax[0, 1].scatter(nxt_pts_klt[:, 0], nxt_pts_klt[:, 1], s=3, c="r")
        ax[0, 1].scatter(nxt_pts_nn[:, 0], nxt_pts_nn[:, 1], s=3, c="g")
        ax[0, 1].scatter(nxt_pts_lkr[:, 0], nxt_pts_lkr[:, 1], s=3, c="y")

        ax[1, 0].clear()
        ax[1, 0].imshow((im1_cv.astype(np.float32) * 0.5 + im2_cv.astype(np.float32) * 0.5).astype(np.uint8),
                        cmap='gray')
        for i in range(0, len(nxt_pts_klt)):
            ax[1, 0].annotate("", xy=nxt_pts_klt[i], xytext=cur_pts_filtered[i],
                              arrowprops=dict(arrowstyle="->", color="r", linewidth=1))

        for i in range(0, len(nxt_pts_nn)):
            ax[1, 0].annotate("", xy=nxt_pts_nn[i], xytext=cur_pts[i],
                              arrowprops=dict(arrowstyle="->", color="g", linewidth=1))

        for i in range(0, len(nxt_pts_lkr)):
            ax[1, 0].annotate("", xy=nxt_pts_lkr[i], xytext=cur_pts[i],
                              arrowprops=dict(arrowstyle="->", color="y", linewidth=1))

        ax[1, 1].clear()
        # ax[1, 1].imshow(visualize.flow_to_img(flow))
        # ax[1, 1].imshow(np.linalg.norm(flow, axis=2) * 10)
        ax[1, 1].imshow(fl.flow_to_image(flow))
        subsample_every_N = 20
        quiver_X = np.arange(0, im1_cv.shape[0], subsample_every_N)
        quiver_Y = np.arange(0, im1_cv.shape[1], subsample_every_N)
        # mesh = np.meshgrid(quiver_X, quiver_Y)
        flow_subsampled = flow[::subsample_every_N, ::subsample_every_N]
        quiver_U = flow_subsampled[:, :, 0]
        quiver_V = -flow_subsampled[:, :, 1]
        ax[1, 1].quiver(quiver_Y, quiver_X, quiver_U, quiver_V)

        ax[0, 0].set_xlim(0, flow.shape[1])
        ax[0, 0].set_ylim(0, flow.shape[0])
        ax[0, 0].invert_yaxis()

        plt.gcf().suptitle(plt_name)

        plt.draw()
        plt.pause(0.001)
        blocking = BlockingKeyInput(fig=plt.gcf())
        key = blocking(timeout=-1)

        return key


c = Compare()

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

# directory = "/home/cs4li/Dev/EUROC/MH_04_difficult/mav0/cam0/data"
# directory = "/home/cs4li/Dev/EUROC/V2_03_difficult/mav0/cam0/data"
# directory = "/home/cs4li/Dev/EUROC/V2_02_medium/mav0/cam0/data"
directory = "/home/cs4li/Dev/TUMVIO/dataset-corridor1_512_16/mav0/cam0/data"
# directory = "/home/cs4li/Dev/TUMVIO/dataset-corridor1_512_16/mav0/cam0/data"
# directory = "/home/cs4li/Dev/TUMVIO/dataset-outdoors1_512_16/mav0/cam0/data"
# directory = "/home/cs4li/Dev/TUMVIO/dataset-outdoors2_512_16/mav0/cam0/data"
# directory = "/home/cs4li/Dev/UZH_FPV/indoor_45_13_snapdragon_with_gt/img"
# directory = "/home/cs4li/Dev/UZH_FPV/outdoor_45_1_snapdragon_with_gt/img"
# directory = "/home/cs4li/Dev/UZH_FPV/indoor_45_9_snapdragon_with_gt/img"
every_N_frames = 1
imgs = natsort.natsorted(os.listdir(directory))

prev_img = None
prev_img_name = ""
prev_img_idx = 0
i = 0

while i < len(imgs):
    if prev_img is not None:
        im = skimage.io.imread(os.path.join(directory, imgs[i]))
        # im = Image.open(os.path.join(directory, imgs[i])).convert("RGB")
        print("reading " + os.path.join(directory, imgs[i]))
        plt_name = "%s: %s -> %s, %s -> %s" % (directory, prev_img_name, imgs[i], prev_img_idx, i)
        key = c.compare_optical_flow(plt_name, prev_img, im, fig, ax)
        prev_img = im
        prev_img_name = imgs[i]
        prev_img_idx = i
        if key == "v":
            i += 2
        elif key == "b":
            i += 3
        elif key == "m":
            i += 10
            print("Skip 10")
        elif key == ",":
            i += 100
            print("Skip 100")
        elif key == ".":
            i += 1000
            print("Skip 1000")
        elif key == "/":
            i += 10000
            print("Skip 10000")
        else:
            i += every_N_frames

    if prev_img is None:
        prev_img = skimage.io.imread(os.path.join(directory, imgs[i]))
        # prev_img = Image.open(os.path.join(directory, imgs[i])).convert("RGB")
        prev_img_name = imgs[i]
        prev_img_idx = i
        i += every_N_frames
