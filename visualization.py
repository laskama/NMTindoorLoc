import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from matplotlib.image import AxesImage


class FloorplanPlot:
    def __init__(self, floorplan_dimensions, floorplan_bg_img="",
                 background_alpha=0.3, sub_plots=None, prreset_fig=None):

        self.floorplan_dimensions = floorplan_dimensions
        self.floorplan_bg_image = floorplan_bg_img

        self.axis: Axes = None
        self.fig: Figure = None

        self.bg_img: AxesImage = None
        self.bg_alpha = background_alpha
        self.sub_plots = sub_plots
        self.preset_fig = prreset_fig

        self.draw_background()

    def show_plot(self):
        plt.show()

    def init_plot(self):
        if self.sub_plots is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = self.preset_fig
            ax = plt.subplot(*self.sub_plots)

        self.axis = ax
        self.fig = fig

    def set_title(self, title="title"):
        self.axis.set_title(title)

    def draw_background(self):
        if self.axis is None:
            self.init_plot()
        # bg image computations
        try:
            bg_image = plt.imread(self.floorplan_bg_image)
            self.bg_img = self.axis.imshow(bg_image, extent=[0, 0 + self.floorplan_dimensions[0],
                                                             0, 0 + self.floorplan_dimensions[1]],
                                           alpha=self.bg_alpha)
        except FileNotFoundError:
            print("No background image found")

    def draw_points(self, x_points, y_points, color='b', alpha=1, **kwargs):
        if self.axis is None:
            self.init_plot()
        # plot raw points
        self.axis.scatter(x_points, y_points, color=color, alpha=alpha, **kwargs)


def visualize_data_time_distribution(acc, gyro, mag, wlan):

    plt.figure()

    plt.scatter(acc['time'], np.ones(len(acc)))
    plt.scatter(gyro['time'], 2 * np.ones(len(gyro)))
    plt.scatter(mag['time'], 3 * np.ones(len(mag)))

    plt.scatter(wlan['time'], 4 * np.ones(len(wlan)))
    plt.show()


def visualize_predictions(true, pred, seq_to_point=False, draw_individually=False):
    floor_dims = pd.read_csv("giaIndoorLoc/floor_dimensions.csv").iloc[1, 1:]

    for b_idx in range(len(true)):

        fp = FloorplanPlot(floor_dims, floorplan_bg_img="giaIndoorLoc/floor_1/floorplan.jpg")

        if seq_to_point:
            fp.draw_points(pred[b_idx, 0], pred[b_idx, 1], color='red')
            fp.draw_points(true[b_idx, 0], true[b_idx, 1], color='green')
        else:
            if draw_individually:
                for s_idx in range(np.shape(pred)[1]):
                    fp.draw_points(pred[b_idx, s_idx, 0], pred[b_idx, s_idx, 1], color='red')
                    fp.draw_points(true[b_idx, s_idx, 0], true[b_idx, s_idx, 1], color='green')
            else:
                fp.draw_points(pred[b_idx, :, 0], pred[b_idx, :, 1], color='red')
                fp.draw_points(true[b_idx, :, 0], true[b_idx, :, 1], color='green')

        fp.show_plot()


def visualize_window_trajectories(pos):
    floor_dims = pd.read_csv("giaIndoorLoc/floor_dimensions.csv").iloc[1, 1:]

    for traj in pos:
        fp = FloorplanPlot(floor_dims, floorplan_bg_img="giaIndoorLoc/floor_1/floorplan.jpg")
        fp.draw_points(traj[:, 0], traj[:, 1])
        fp.show_plot()