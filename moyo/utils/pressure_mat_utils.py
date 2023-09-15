import re
import c3d
import cv2
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs

VICON_FPS = 60
PRESSURE_MAP_FPS = 60
IOI_FPS = 30

class PressureMat:
    def __init__(self, csv_path, xml_path, c3d_file):
        self.csv_path = csv_path
        self.xml_path = xml_path
        self.c3d_file = c3d_file

        self.marker_positions_metric = self.extract_pressure_mat_markers()

        self.gt_cops_relative, gt_pressure_frameids = self.process_csv()
        # self.gt_cops_metric = self.compute_gt_cop_metric()
        self.gt_pressures, self.gt_heatmaps, self.mat_size, self.mat_size_metric = self.process_xml()
        # filter out pressure frames that are not in the csv file
        self.gt_pressures = self.gt_pressures[gt_pressure_frameids]
        self.gt_heatmaps = [self.gt_heatmaps[i] for i in gt_pressure_frameids]


    def parse_pressure_crops(self, pressure_crop):
        '''
            Takes raw pressure data from xml which is a list of strings for every frame containing pressure
            pressure_crops: N x 1 list of strings
            '''
        pressure_crop = [pressure_crop[i].get_text() for i in range(len(pressure_crop))]
        pressure_crop = [re.split(r'\n\t+', pressure_crop[i])[1:-1] for i in range(len(pressure_crop))]
        for i in range(len(pressure_crop)):
            for j in range(len(pressure_crop[i])):
                row = re.split(' ', pressure_crop[i][j])
                row = [float(el) for el in row]
                pressure_crop[i][j] = row
        return pressure_crop

    def unnormalize_pressure(self, pressure_crop, cell_begin, cell_end, mat_size, frame_count):
        '''
        Unnormalize pressure data to the full size of the pressure mat
        pressure_crop: N x 1 list of lists of lists (N frames, 1 pressure crop, pressure_crop[i] is a list of lists of floats (pressure values))
        cell_begin: N x 2 list of lists (N frames, 2 coordinates of the beginning of the pressure crop)
        cell_end: N x 2 list of lists (N frames, 2 coordinates of the end of the pressure crop)
        mat_size: 2 x 1 list (2 coordinates of the full size of the pressure mat)
        '''
        pressure_unnorm = np.zeros([frame_count, mat_size[1], mat_size[0]])
        for i in range(frame_count):
            pressure_crop_i = np.flipud(pressure_crop[i])
            pressure_unnorm[i, cell_begin[i][1]:cell_end[i][1],
            cell_begin[i][0]:cell_end[i][0]] = pressure_crop_i
        return pressure_unnorm

    def process_xml(self):
        # read xml file and parse
        with open(self.xml_path, 'r') as f:
            xml = f.read()
        xml_soup = bs(xml, 'lxml')
        mat_size = [int(xml_soup.movements.clips.clip.cell_count.x.get_text()),
                    int(xml_soup.movements.clips.clip.cell_count.y.get_text())]  # number of cells in x and y direction
        sensel_size = [float(xml_soup.movements.clips.clip.cell_size.x.get_text()),
                       float(xml_soup.movements.clips.clip.cell_size.y.get_text())]  # size of every cell in mm
        mat_size_metric = [mat_size[0] * sensel_size[0] / 1000,
                           mat_size[1] * sensel_size[1] / 1000]  # size of the pressure mat in m
        fps = float(xml_soup.movements.clips.clip.frequency.get_text())
        frame_count = int(xml_soup.movements.clips.clip.count.get_text())

        # the pressure mat crops the pressure data, uncrop it to the full size
        cell_begin_raw = xml_soup.movements.clips.clip.data.find_all('cell_begin')
        assert len(
            cell_begin_raw) == frame_count, 'Number of frames in xml file does not match the number of frames in the pressure data'
        cell_begin = [[int(cell_begin_raw[i].x.get_text()), int(cell_begin_raw[i].y.get_text())] for i in
                      range(len(cell_begin_raw))]
        cell_extent_raw = xml_soup.movements.clips.clip.data.find_all('cell_count')
        assert len(
            cell_extent_raw) == frame_count, 'Number of frames in xml file does not match the number of frames in the pressure data'
        cell_end = [[cell_begin[i][0] + int(cell_extent_raw[i].x.get_text()),
                     cell_begin[i][1] + int(cell_extent_raw[i].y.get_text())] for i in range(len(cell_extent_raw))]
        pressure_crop_raw = xml_soup.movements.clips.clip.data.find_all('cells')
        assert len(
            pressure_crop_raw) == frame_count, 'Number of frames in xml file does not match the number of frames in the pressure data'
        # parse pressure crops
        pressure_crop = self.parse_pressure_crops(pressure_crop_raw)
        # unnormalize pressure crops
        pressure_unnorm = self.unnormalize_pressure(pressure_crop, cell_begin, cell_end, mat_size, frame_count)
        # visualize pressure as heatmap
        heatmaps = vis_heatmap_seq(pressure_unnorm, normalize=True)
        return pressure_unnorm, heatmaps, mat_size, mat_size_metric


    def process_csv(self):
        # read csv file and parse
        pressure_md = pd.read_csv(self.csv_path, header=[0], nrows=1)
        pressure_df = pd.read_csv(self.csv_path, header=[2])

        # extract important pressure md from csv
        pressure_fps = pressure_md['frequency'].values[0]
        assert pressure_fps == PRESSURE_MAP_FPS, 'Pressure fps is not equal to default'
        pressure_frame_count = pressure_md['count'].values[0]

        # get the cop coordinates
        cop_x = pressure_df['Pressure, Raw Pressure-distribution-Pressure center-x (mm)'].values / 1000  # mm to m
        cop_y = pressure_df['Pressure, Raw Pressure-distribution-Pressure center-y (mm)'].values / 1000
        cop = np.array([cop_x, cop_y]).T

        pressure_timestamps = pressure_df['time']
        # convert pressure df to numpy array
        pressure_timestamps = pressure_timestamps.values
        assert pressure_timestamps.shape[
                   0] == pressure_frame_count, 'Number of frames in md does not match the number of frames in the timestamp'

        # convert timestamps to frame numbers and round to lower
        pressure_fids = np.rint(pressure_timestamps * pressure_fps).astype(int)
        assert pressure_fids.shape[0] == pressure_fids[-1] - pressure_fids[
            0] + 1, 'Some frames are missing in the timestamps'
        return cop, pressure_fids


    def extract_pressure_mat_markers(self):
        with open(self.c3d_file, 'rb') as f:
            data = c3d.Reader(f)  # data contains the header of the c3d files, info such as the label name
            for frame in data.read_frames():  # call read_frames() to iterate across the marker positions
                frame = frame[1][:, :3]  # this contains the marker positions
                break

        marker_id_one = np.where(data.point_labels == 'SensorMat:M_1                                                   ')[0][0]
        marker_id_two = np.where(data.point_labels == 'SensorMat:M_2                                                   ')[0][0]
        marker_id_three = np.where(data.point_labels == 'SensorMat:M_3                                                   ')[0][0]
        marker_id_four = np.where(data.point_labels == 'SensorMat:M_4                                                   ')[0][0]
        marker_ids_unordered = [marker_id_one, marker_id_two, marker_id_three, marker_id_four]

        # Put the markers in the correct order
        for i, marker_id in enumerate(marker_ids_unordered):
            assert marker_id is not None, f'Marker {i} not found'
            if frame[marker_id, :][0] > 0 and frame[marker_id, :][1] > 0:
                marker_id_bl = marker_id
            if frame[marker_id, :][0] > 0 and frame[marker_id, :][1] < 0:
                marker_id_tl = marker_id
            if frame[marker_id, :][0] < 0 and frame[marker_id, :][1] < 0:
                marker_id_tr = marker_id
            if frame[marker_id, :][0] < 0 and frame[marker_id, :][1] > 0:
                marker_id_br = marker_id

        marker_ids = [marker_id_bl, marker_id_tl, marker_id_tr, marker_id_br]      # (bottom left, top left, top right, bottom right)

        marker_positions = frame[marker_ids, :]
        return marker_positions

    # def compute_gt_cop_metric(self):
    #     """
    #     Convert relative gt_cop to metric gt_cop in the world coordinates
    #     Args:
    #         gt_cop: relative to the pressure mat
    #         mat_size_metric:
    #         marker_positions:
    #
    #     Returns:
    #
    #     """
    #     import ipdb; ipdb.set_trace()
    #     mat_marker_bl = self.marker_positions_metric[0]
    #     mat_marker_br = self.marker_positions_metric[1]
    #
    #     # convert gt_cop to metric
    #     gt_cop_metric = self.gt_cops_relative.copy()
    #     gt_cop_metric[:, 0] = mat_marker_br[0] / 1000 - self.gt_cops_relative[:,
    #                                                     0]  # because the x-axis is to your right in mat coordinate but to your left in image coordinates
    #     gt_cop_metric[:, 1] = self.gt_cops_relative[:, 1] + mat_marker_bl[1] / 1000
    #     return gt_cop_metric


def vis_heatmap_seq(pressure, normalize=True):
    # normalize pressure value over all frames
    # if normalize:
    #     pressure = (pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure))
    # visualize pressure as heatmap
    heatmaps = []
    for i in range(len(pressure)):
        if normalize:
            pressure_i = (pressure[i] - np.min(pressure[i])) / (np.max(pressure[i]) - np.min(pressure[i]))
        else:
            pressure_i = pressure[i]
        heatmap = cv2.applyColorMap(np.uint8(255 * pressure_i), cv2.COLORMAP_JET)
        heatmaps.append(heatmap)
    return heatmaps


def vis_heatmap(pressure, normalize=True):
    if normalize:
        pressure = (pressure - np.min(pressure)) / (np.max(pressure) - np.min(pressure))
    heatmap = cv2.applyColorMap(np.uint8(255 * pressure), cv2.COLORMAP_JET)
    return heatmap