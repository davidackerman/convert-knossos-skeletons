from collections import OrderedDict
from functools import cached_property
from scipy.interpolate import splprep, splev
import networkx as nx
from scipy.optimize import fmin
from typing import List
from scipy.spatial import distance_matrix
import numpy as np
from fitters import *


def add_to_dict(key_value_dict, key, value):
    if key not in key_value_dict:
        key_value_dict[key] = np.array([value])
    else:
        key_value_dict[key] = np.append(key_value_dict[key], value)


class Filaments:
    def __init__(self, bin_size, splint_fraction=0.01):
        self.splint_fraction = splint_fraction
        self.num_points = 1 / splint_fraction + 1
        self.bin_size = bin_size
        self.filament_list: List[Filament] = list()

    def add_filament(self, id, node_dic, edge_list):
        self.filament_list.append(
            Filament(id, node_dic, edge_list, self.bin_size, self.splint_fraction)
        )

    def combine_filaments_info(self, indices="all"):
        self.bin_to_cos_theta_dict = {}
        self.bin_to_R_squared_dict = {}
        self.bin_to_delta_squared_dict = {}
        if indices == "all":
            indices = range(0, len(self.filament_list))
        for index in indices:
            current_filament = self.filament_list[index]
            for key, value in current_filament.bin_to_cos_theta_dict.items():
                add_to_dict(self.bin_to_cos_theta_dict, key, value)
            for key, value in current_filament.bin_to_R_squared_dict.items():
                add_to_dict(self.bin_to_R_squared_dict, key, value)
            for key, value in current_filament.bin_to_delta_squared_dict.items():
                add_to_dict(self.bin_to_delta_squared_dict, key, value)

        self.cos_theta = []
        self.R_squared = []
        self.delta_squared = []
        for index in indices:
            current_filament = self.filament_list[index]
            self.cos_theta.append(current_filament.cos_theta)
            self.R_squared.append(current_filament.R_squared)
            self.delta_squared.append(current_filament.delta_squared)

        self.cos_theta = np.vstack(self.cos_theta)
        self.R_squared = np.vstack(self.R_squared)
        self.delta_squared = np.vstack(self.delta_squared)

    @cached_property
    def point_to_point_info(self):
        dic = {}
        for i in range(len(self.filament_list)):
            filament_i = self.filament_list[i]
            for j in range(i + 1, len(self.filament_list)):
                filament_j = self.filament_list[j]

                dist = distance_matrix(filament_i.coords, filament_j.coords)
                # index and values
                closest_point_i_per_point_j = np.argmin(dist, axis=0)
                closest_point_j_per_point_i = np.argmin(dist, axis=1)

                min_distance_per_point_j = np.amin(dist, axis=0)
                min_distance_per_point_i = np.amin(dist, axis=1)

                abs_cos_theta_per_point_j = np.zeros(closest_point_i_per_point_j.shape)
                abs_cos_theta_per_point_i = np.zeros(closest_point_j_per_point_i.shape)
                for point_j, point_i in enumerate(closest_point_i_per_point_j):
                    cos_theta = np.dot(
                        filament_i.normalized_tangents[point_i, :],
                        filament_j.normalized_tangents[point_j, :],
                    )
                    abs_cos_theta_per_point_j[point_j] = np.abs(cos_theta)
                for point_i, point_j in enumerate(closest_point_j_per_point_i):
                    cos_theta = np.dot(
                        filament_i.normalized_tangents[point_i, :],
                        filament_j.normalized_tangents[point_j, :],
                    )
                    abs_cos_theta_per_point_i[point_i] = np.abs(cos_theta)

                dic[(i, j)] = (
                    min_distance_per_point_j,
                    abs_cos_theta_per_point_j,
                )
                dic[(j, i)] = (
                    min_distance_per_point_i,
                    abs_cos_theta_per_point_i,
                )

        return dic

    def calculate_bundles(
        self, cutoff_distance=20, cutoff_angle=45, cutoff_fraction=0.25
    ):
        cutoff_cos_theta = np.abs(np.cos(np.pi * cutoff_angle / 180))
        self.are_bundled = np.zeros((len(self.filament_list), len(self.filament_list)))

        for i in range(len(self.filament_list)):
            for j in range(i + 1, len(self.filament_list)):
                (
                    min_distance_per_point_j,
                    abs_cos_theta_per_point_j,
                ) = self.point_to_point_info[(i, j)]

                (
                    min_distance_per_point_i,
                    abs_cos_theta_per_point_i,
                ) = self.point_to_point_info[(j, i)]

                # count_nonzero/count seems faster than np.mean
                fraction_i = (
                    np.count_nonzero(
                        (min_distance_per_point_i <= cutoff_distance)
                        & (abs_cos_theta_per_point_i >= cutoff_cos_theta)
                    )
                    / self.num_points
                )

                fraction_j = (
                    np.count_nonzero(
                        (min_distance_per_point_j <= cutoff_distance)
                        & (abs_cos_theta_per_point_j >= cutoff_cos_theta)
                    )
                    / self.num_points
                )

                if fraction_i >= cutoff_fraction:
                    self.are_bundled[i, j] = 1
                if fraction_j >= cutoff_fraction:
                    self.are_bundled[j, i] = 1

    def calculate_bundle_properties(
        self, cutoff_distance=20, cutoff_angle=45, cutoff_fraction=0.25
    ):
        self.calculate_bundles(cutoff_distance, cutoff_angle, cutoff_fraction)
        self.bundles = []
        num_bundled = self.are_bundled.sum(axis=1)
        unique_num_bundled = np.unique(num_bundled)
        for current_num_bundled in unique_num_bundled:
            indices = np.argwhere(num_bundled == current_num_bundled).flatten()
            bundle_filament_list = [self.filament_list[i] for i in indices]
            self.bundles.append(Bundle(bundle_filament_list, current_num_bundled))


class Bundle:
    def __init__(self, filament_list, num_bundled, bin_size=50):
        self.filament_list = filament_list
        self.num_bundled = int(num_bundled)
        self.calculate_properties()
        self.fit_curves()

    def calculate_properties(self):
        # self.cos_theta = []
        # self.R_squared = []
        # self.delta_squared = []
        # for current_filament in self.filament_list:
        #     self.cos_theta.append(current_filament.cos_theta)
        #     self.R_squared.append(current_filament.R_squared)
        #     self.delta_squared.append(current_filament.delta_squared)

        # self.cos_theta = np.vstack(self.cos_theta)
        # self.R_squared = np.vstack(self.R_squared)
        # self.delta_squared = np.vstack(self.delta_squared)

        # np.histogram(self.cos_theta,np)

        self.bin_to_cos_theta_dict = {}
        self.bin_to_R_squared_dict = {}
        self.bin_to_delta_squared_dict = {}
        for current_filament in self.filament_list:
            for key, value in current_filament.bin_to_cos_theta_dict.items():
                add_to_dict(self.bin_to_cos_theta_dict, key, value)
            for key, value in current_filament.bin_to_R_squared_dict.items():
                add_to_dict(self.bin_to_R_squared_dict, key, value)
            for key, value in current_filament.bin_to_delta_squared_dict.items():
                add_to_dict(self.bin_to_delta_squared_dict, key, value)

        for key, value in self.bin_to_cos_theta_dict.items():
            self.bin_to_cos_theta_dict[key] = np.mean(value)
        for key, value in self.bin_to_R_squared_dict.items():
            self.bin_to_R_squared_dict[key] = np.mean(value)
        for key, value in self.bin_to_delta_squared_dict.items():
            self.bin_to_delta_squared_dict[key] = np.mean(value)

        self.bin_to_cos_theta_dict = dict(sorted(self.bin_to_cos_theta_dict.items()))
        self.bin_to_R_squared_dict = dict(sorted(self.bin_to_R_squared_dict.items()))
        self.bin_to_delta_squared_dict = dict(
            sorted(self.bin_to_delta_squared_dict.items())
        )

    def fit_curves(self, stop_nm=500, start_nm=0):
        self.P_cos_theta = fit_func(
            cos_theta_equation, self.bin_to_cos_theta_dict, stop_nm, start_nm
        )
        self.P_R_squared = fit_func(
            R_squared_equation, self.bin_to_R_squared_dict, stop_nm, start_nm
        )
        self.P_delta_squared = fit_func(
            delta_squared_equation, self.bin_to_delta_squared_dict, stop_nm, start_nm
        )


class Filament:
    def __init__(self, id, node_dic, edge_list, bin_size, splint_fraction=0.01):
        self.splint_fraction = splint_fraction
        self.id = id
        self.bin_size = bin_size
        self.get_coords_tangents_and_tck(node_dic, edge_list)
        self.get_lengths()
        self.get_cos_theta()
        self.get_R_squared()
        self.get_delta_squared()

    def get_lengths(self):
        vectors = self.coords[1:, :] - self.coords[:-1, :]
        lengths = np.sum(vectors**2, axis=1) ** 0.5
        lengths = np.insert(lengths, 0, 0)
        self.cumsum_lengths = np.cumsum(lengths)

    def node_edges_to_xyz(self, node_dic, edge_list):
        g = nx.from_edgelist(edge_list)
        xyz = []
        for node in g.nodes():
            xyz.append(node_dic[node] * 2)  # voxel-size = 2
        xyz = np.array(xyz)
        return xyz[:, 0], xyz[:, 1], xyz[:, 2]

    def get_coords_tangents_and_tck(self, node_dic, edge_list):
        x, y, z = self.node_edges_to_xyz(node_dic, edge_list)
        k = 3  # default is k=3, but this gave things that shot off to 1E6 sometimes
        if len(x) <= 3:
            k = len(x) - 1
        self.tck, _ = splprep([x, y, z], k=k, s=0)

        self.unew = np.arange(0, 1.00 + self.splint_fraction, self.splint_fraction)
        new_x, new_y, new_z = splev(self.unew, self.tck)
        d_x, d_y, d_z = splev(self.unew, self.tck, der=1)
        tangents = np.column_stack((d_x, d_y, d_z))

        self.coords = np.column_stack((new_x, new_y, new_z))
        self.num_points = self.coords.shape[0]
        self.normalized_tangents = tangents / (
            np.linalg.norm(tangents, axis=1)[:, None]
        )

    def get_cos_theta(self):
        self.bin_to_cos_theta_dict = {}
        self.cos_theta = np.zeros((np.sum(range(self.num_points)), 2))
        count = 0
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                l = self.cumsum_lengths[j] - self.cumsum_lengths[i]
                bin = (l // self.bin_size + 0.5) * self.bin_size
                cos_theta = np.dot(
                    self.normalized_tangents[i], self.normalized_tangents[j]
                )
                self.cos_theta[count, :] = [l, cos_theta]
                add_to_dict(self.bin_to_cos_theta_dict, bin, cos_theta)
                count += 1

    def get_R_squared(self):
        self.bin_to_R_squared_dict = {}
        self.R_squared = np.zeros((np.sum(range(self.num_points)), 2))
        count = 0
        for i in range(self.num_points):
            for j in range(i + 1, self.num_points):
                l = self.cumsum_lengths[j] - self.cumsum_lengths[i]
                bin = (l // self.bin_size + 0.5) * self.bin_size
                R_squared = np.linalg.norm(self.coords[i] - self.coords[j]) ** 2
                self.R_squared[count, :] = [l, R_squared]
                add_to_dict(self.bin_to_R_squared_dict, bin, R_squared)
                count += 1

    def get_delta_squared(self):
        def distance_to_spline(u, tck, point):
            point_on_spline = np.array(splev(u, tck)).flatten()
            return np.linalg.norm(point_on_spline - point)

        self.bin_to_delta_squared_dict = {}
        knots = np.unique(self.tck[0])
        # knots = self.unew
        knot_points = np.array(splev(knots, self.tck))
        self.delta_squared = np.zeros((np.sum(range(len(knots))), 2))
        count = 0
        for i in range(len(knots)):
            for j in range(i + 1, len(knots)):
                # secant midpoint
                points = knot_points[:, [i, j]]
                L = np.linalg.norm(points[:, 0] - points[:, 1])
                point = np.array(points).mean(axis=1)
                u_guess = (knots[i] + knots[j]) / 2
                res = fmin(
                    distance_to_spline,
                    u_guess,
                    (self.tck, point),
                    full_output=True,
                    disp=False,
                )
                bin = (L // self.bin_size + 0.5) * self.bin_size

                delta_squared = res[1] ** 2
                self.delta_squared[count, :] = [L, delta_squared]
                add_to_dict(self.bin_to_delta_squared_dict, bin, delta_squared)
                count += 1
