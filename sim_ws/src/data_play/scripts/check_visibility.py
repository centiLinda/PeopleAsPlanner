#!/usr/bin/env python3

import sys
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import numpy as np
import pickle
from typing import List

def save_data(data, file_path):
    """ Save data to file with suffix .pkl"""
    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()
    return True

def read_data(file_path):
    with open(file_path, 'rb') as file:
        data =pickle.load(file)
        return data
    
def generate_circle_points(radius, numPoints) -> np.ndarray:
    points = []
    for i in range(numPoints):
        theta = 2 * math.pi * i / numPoints
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append([x, y])
    return np.array(points)

def spherical_flipping(flip_radius, point: tuple) -> tuple:
    """ Note the point should be transoformed into local coordinate frame."""
    norm = math.sqrt(point[0]**2 + point[1]**2)
    return 2*flip_radius*point[0]/norm - point[0], 2*flip_radius*point[1]/norm - point[1]

def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta

def interpolate_points_with_numSteps(pt1, pt2, num_steps) -> np.ndarray:
    point1 = np.array(pt1)
    point2 = np.array(pt2)

    distance = np.linalg.norm(point2 - point1)
    direction = (point2 -point1) / distance
    step = round(distance / num_steps, 6)
    if step == 0:
        return np.array([point1, point2])

    inter_points = [np.array((point1[0] + direction[0] * step * i, point1[1] + direction[1] * step * i)) for i in range(0, num_steps)]
    inter_points.append(point2)
    return np.array(inter_points)

def interpolate_points_with_step(pt1, pt2, step):
    points = [pt1]

    distance = ((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2) ** 0.5
    if distance <= step:
        points.append(pt2)
        return points

    direction = [(pt2[0] - pt1[0]) / distance, (pt2[1] - pt1[1]) / distance]
    num_steps = int(distance // step)

    for i in range(1, num_steps + 1):
        new_point = (
            pt1[0] + direction[0] * step * i,
            pt1[1] + direction[1] * step * i
        )
        points.append(new_point)

    points.append(pt2)
    return points

def construct_hull_with_small_faces(hull_faces: np.ndarray, angle_step: float):
    hull_small_faces = []
    for k in range(hull_faces.shape[0]):
        face = hull_faces[k]
        rho1, theta1 = cartesian_to_polar(face[0][0], face[0][1])
        rho2, theta2 = cartesian_to_polar(face[1][0], face[1][1])
        num_steps = round(abs(theta1 - theta2) / (angle_step/180*np.pi))
        if num_steps > 1:
            small_face_vertices = interpolate_points_with_numSteps(face[0], face[1], num_steps)
            for k in range(len(small_face_vertices)-1):
                hull_small_faces.append([small_face_vertices[k], small_face_vertices[k+1]])
        else:
            hull_small_faces.append(face)
    return np.array(hull_small_faces)

def flip_hull_faces(hull_faces, flip_radius):
    return 2*flip_radius*hull_faces/np.linalg.norm(hull_faces, axis=2, keepdims=True) - hull_faces

def get_all_dist_and_gradient(faces: np.ndarray, robot_pose: np.ndarray):
    """ faces: (n, 2); robot_pose: (2,) """
    dim = len(faces[0][0])
    X0 = robot_pose[:dim].reshape(1, -1)  # (1, 2)
    A = faces[:, 0] - faces[:, 1]   # (n, 2)
    B = X0 - faces[:, 1]  # (n, 2)
    t = np.sum(A * B, axis=1) / np.sum(A * A, axis=1) # (n,)

    mask_t_le_0 = t <= 0
    mask_t_ge_1 = t >= 1
    mask_t_between = (t > 0) & (t < 1)
    
    los_dist = np.zeros(len(faces))
    gradient = np.zeros((len(faces), dim))

    if np.any(mask_t_le_0):
        los_dist[mask_t_le_0] = np.linalg.norm(X0 - faces[mask_t_le_0, 1], axis=1)
        gradient[mask_t_le_0] = (1 / (2 * los_dist[mask_t_le_0]))[:, None] * 2 * (X0 - faces[mask_t_le_0, 1])

    if np.any(mask_t_ge_1):
        los_dist[mask_t_ge_1] = np.linalg.norm(X0 - faces[mask_t_ge_1, 0], axis=1)
        gradient[mask_t_ge_1] = (1 / (2 * los_dist[mask_t_ge_1]))[:, None] * 2 * (X0 - faces[mask_t_ge_1, 0])
    
    if np.any(mask_t_between):
        A_bet = A[mask_t_between]
        B_bet = B[mask_t_between]
        los_dist[mask_t_between] = np.sqrt(-np.sum(A_bet * B_bet, axis=1)**2/np.sum(A_bet * A_bet, axis=1) + np.sum(B_bet * B_bet, axis=1))
        gradient[mask_t_between] = (1 / (2 * los_dist[mask_t_between]))[:, None] * (-np.sum(A_bet * B_bet, axis=1)[:, None] / np.sum(A_bet * A_bet, axis=1)[:, None] * 2 * A_bet + 2 * B_bet)
    return los_dist, gradient

def from_scan_to_pointCloud(laser_scan, angle_min=-3.141590118408203, angle_incre=0.008738775737583637, max_range=30, human_list: np.ndarray=np.array([])) -> np.ndarray:
    """human_list: around which point cloud should be removed to avoid repeated consideration."""
    new_scan = []  # scan with 2D coordinates
    angle_incre = angle_incre
    curr_angle = angle_min
    skip_count = 0

    for data in laser_scan:
        if skip_count < 8:
            curr_angle += angle_incre 
            skip_count += 1
            continue
        skip_count = 0  
            
        if data >= max_range:
            data = max_range
        if data <= 0.1:
            curr_angle += angle_incre 
            continue
        new_data = (data*math.cos(curr_angle), data*math.sin(curr_angle))
        curr_angle += angle_incre
        if human_list.size != 0:
            pose_diff = np.array(new_data).reshape(1, -1) - human_list
            dists = np.linalg.norm(pose_diff, axis=1).reshape(-1)
            if np.any(dists < 1.5):    # dist is less than 1.5 meter, then remove this
                continue
        new_scan.append(new_data)
    
    # Augment point cloud to fill in gaps
    num_add_point = 0 if len(new_scan) > 10 else 100
    circle_points = generate_circle_points(max_range, num_add_point)

    if len(new_scan) == 0:
        # print(f"### New_scan is empty! #laser_scan{len(laser_scan)}##")
        return circle_points

    if circle_points.size == 0:
        return np.array(new_scan)

    final_pCloud = np.vstack((np.array(new_scan), circle_points))

    return final_pCloud

def transform_point_to_local_frame(robot_origin, point) -> np.ndarray:
    """Args:
        robot_origin (List): (x, y, theta)
        point (List): _description_
    """
    angle = -robot_origin[2]
    rotation_R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    abs_position = np.array(point) - np.array(robot_origin[:2])
    return np.dot(rotation_R, abs_position).reshape(-1)

def expand_human_obstacles_to_pCloud(robot_pose: List, local_human_list: List[list], human_radius: List[float]) -> List[np.ndarray]:
    # Transform humans to robot's local coordinate frame
    pCloud_of_humans = []
    for k, human_local in enumerate(local_human_list):
        circle_points = generate_circle_points(human_radius[k], 30)
        this_human_points = human_local + circle_points
        pCloud_of_humans.append(this_human_points)
    return pCloud_of_humans

def from_pCloud_to_visibleRegion(pointCloud: np.ndarray, local_test_point: np.ndarray, flip_radius: float):
    flipped_scan = []
    for data in pointCloud:
        if np.array_equal(data, np.zeros(2)):
            continue
        flipped_data = spherical_flipping(flip_radius, tuple(data))
        flipped_scan.append(flipped_data)

    hull = ConvexHull(np.array(flipped_scan))
    hull_vertices_index = hull.vertices

    hull_faces = []
    normal_vectors = []
    vertices_on_hull = []

    for i in range(hull_vertices_index.shape[0]):
        idx1, idx2 = hull_vertices_index[i% hull_vertices_index.shape[0]], hull_vertices_index[(i+1) % hull_vertices_index.shape[0]]
        hull_faces.append([flipped_scan[idx1], flipped_scan[idx2]])
        vertices_on_hull.append(flipped_scan[idx1])
        # Calculate normal vector
        a = flipped_scan[idx1][1] - flipped_scan[idx2][1]
        b = -(flipped_scan[idx1][0] - flipped_scan[idx2][0])
        n_k = np.array((a, b))
        n_k = n_k / np.linalg.norm(n_k)
        # Here we should check whether the normal vector points to outwards or inwards
        if np.dot(n_k, (flipped_scan[idx1][0], flipped_scan[idx1][1])) < 0:
            n_k = -n_k
        normal_vectors.append(n_k)

    hull_faces = np.array(hull_faces)
    normal_vectors = np.array(normal_vectors)
    vertices_on_hull = np.array(vertices_on_hull)

    hull_with_small_faces = construct_hull_with_small_faces(hull_faces, angle_step=0.5)
    flipped_small_faces = flip_hull_faces(hull_with_small_faces, flip_radius)
    return flipped_small_faces

def check_visibility(pointCloud: np.ndarray, local_test_point: np.ndarray, flip_radius: float):
    flipped_testPoint = spherical_flipping(flip_radius, tuple(local_test_point))

    flipped_scan = []
    for data in pointCloud:
        if np.array_equal(data, np.zeros(2)):
            continue
        flipped_data = spherical_flipping(flip_radius, tuple(data))
        flipped_scan.append(flipped_data)

    hull = ConvexHull(np.array(flipped_scan))
    hull_vertices_index = hull.vertices

    hull_faces = []
    normal_vectors = []
    vertices_on_hull = []

    for i in range(hull_vertices_index.shape[0]):
        idx1, idx2 = hull_vertices_index[i% hull_vertices_index.shape[0]], hull_vertices_index[(i+1) % hull_vertices_index.shape[0]]
        hull_faces.append([flipped_scan[idx1], flipped_scan[idx2]])
        vertices_on_hull.append(flipped_scan[idx1])
        # Calculate normal vector
        a = flipped_scan[idx1][1] - flipped_scan[idx2][1]
        b = -(flipped_scan[idx1][0] - flipped_scan[idx2][0])
        n_k = np.array((a, b))
        n_k = n_k / np.linalg.norm(n_k)
        # Here we should check whether the normal vector points to outwards or inwards
        if np.dot(n_k, (flipped_scan[idx1][0], flipped_scan[idx1][1])) < 0:
            n_k = -n_k
        normal_vectors.append(n_k)
    hull_faces = np.array(hull_faces)
    normal_vectors = np.array(normal_vectors)
    vertices_on_hull = np.array(vertices_on_hull)
    temp_point = np.array(flipped_testPoint).reshape(1, -1)
    dists_to_hull = ((temp_point - vertices_on_hull) * normal_vectors).sum(axis=1).reshape(-1)

    idx_max = np.argmax(dists_to_hull)
    if dists_to_hull[idx_max] < 0:  # test whether point is visible
        return -1

    hull_with_small_faces = construct_hull_with_small_faces(hull_faces, angle_step=0.5)
    flipped_small_faces = flip_hull_faces(hull_with_small_faces, flip_radius)

    los_dists, gradients = get_all_dist_and_gradient(flipped_small_faces, local_test_point)
    idx_min = np.argmin(los_dists)
    return los_dists[idx_min]

def human_scoring(laser_scan: List, human_list: List[list], robot_pose: List, human_radius: List[float], flip_radius: float = 500) -> List:
    """Return visibility of each human.

    Args:
        laser_scan (List): point cloud measurement
        human_list (List[list]): surrounding human positions (world frame)
        robot_pose (List): robot's pose (world frame)
        human_radius (List[float]): expansion of each human. Defaults to 0.6.
        flip_raidus: for visible region construction
    """
    num_human = len(human_list)
    visibility_score = [-1 for _ in range(num_human)]   # -1: no visible; >=0: LoS-distance

    if not is_mutual_dist_safe(human_list, robot_pose, human_radius):
        print("#### Robot fully blocked by neighbor####")

    robot_pose = np.array(robot_pose)
    # Transform humans into robot's local coordinate frame
    local_human_list = []
    for human in human_list:
        human_pose = [human[1], human[2]]
        human_local = transform_point_to_local_frame(robot_pose, human_pose)
        local_human_list.append(human_local)

    pointCloud = from_scan_to_pointCloud(laser_scan, human_list=np.array(local_human_list))
    pCloud_of_humans = expand_human_obstacles_to_pCloud(robot_pose, local_human_list, human_radius)
    index_of_human_pointCloud = []  # start_index, end_index (included)
    for human_pCloud in pCloud_of_humans:
        start_index = len(pointCloud)
        pointCloud = np.vstack((pointCloud, human_pCloud))
        end_index = len(pointCloud) - 1
        index_of_human_pointCloud.append([start_index, end_index])
    
    # Check visibility of each person
    visibility_score = [-1 for _ in range(num_human)]   # -1: no visible; >=0: LoS-distance
    for i in range(num_human):
        thisPointCloud = pointCloud.copy()
        thisPointCloud[index_of_human_pointCloud[i][0]:index_of_human_pointCloud[i][1]+1] = [0, 0]
        this_score = check_visibility(thisPointCloud, local_human_list[i], flip_radius=flip_radius)
        visibility_score[i] = this_score

    return visibility_score

def get_visible_region(laser_scan: List, human_list: List[list], robot_pose: List, human_radius: List[float], flip_radius: float = 500) -> np.ndarray:
    """Return visibility of each human.

    Args:
        laser_scan (List): point cloud measurement
        human_list (List[list]): surrounding human positions (world frame)
        robot_pose (List): robot's pose (world frame)
        human_radius (List[float]): expansion of each human. Defaults to 0.6.
        flip_raidus: for visible region construction
    """
    robot_pose = np.array(robot_pose)
    # Transform humans into robot's local coordinate frame
    local_human_list = []
    for human in human_list:
        human_pose = [human[1], human[2]]
        human_local = transform_point_to_local_frame(robot_pose, human_pose)
        local_human_list.append(human_local)

    # Get point cloud without human
    pointCloud = from_scan_to_pointCloud(laser_scan, human_list=np.array(local_human_list))
    visible_region_edges = from_pCloud_to_visibleRegion(pointCloud, np.array([0, 0]), flip_radius)
    
    return visible_region_edges

def is_mutual_dist_safe(human_list, robot_pose, human_radius):
    for k, human in enumerate(human_list):
        human_pose = [human[1], human[2]]
        dist = np.linalg.norm(np.array(human_pose) - np.array(robot_pose[:2]))
        if dist < 2*human_radius[k]:
            return False

    return True

def visualize_visible_region(laser_scan, human_list, robot_pose, human_radius, flip_radius, target_idx, ax):
    robot_pose = np.array(robot_pose)

    # Transform humans into robot's local coordinate frame
    local_human_list = []
    for human in human_list:
        human_pose = [human[1], human[2]]
        human_local = transform_point_to_local_frame(robot_pose, human_pose)
        local_human_list.append(human_local)

    pointCloud = from_scan_to_pointCloud(laser_scan)
    pCloud_of_humans = expand_human_obstacles_to_pCloud(robot_pose, local_human_list, human_radius)
    for k, human_pCloud in enumerate(pCloud_of_humans):
        if k == target_idx:
            continue
        pointCloud = np.vstack((pointCloud, human_pCloud))
    
    # PointCloud flipping and convexhull construction
    flipped_scan = []
    for data in pointCloud:
        if np.array_equal(data, np.zeros(2)):
            continue
        flipped_data = spherical_flipping(flip_radius, tuple(data))
        flipped_scan.append(flipped_data)

    hull = ConvexHull(np.array(flipped_scan))
    hull_vertices_index = hull.vertices

    hull_faces = []
    for i in range(hull_vertices_index.shape[0]):
        idx1, idx2 = hull_vertices_index[i% hull_vertices_index.shape[0]], hull_vertices_index[(i+1) % hull_vertices_index.shape[0]]
        hull_faces.append([flipped_scan[idx1], flipped_scan[idx2]])
    hull_faces = np.array(hull_faces)

    # Plot convexhull and visible region
    for k in range(len(hull_faces)):
        point1, point2 = hull_faces[k]     
        edge_points = interpolate_points_with_step(point1, point2, step=0.1)
        edge_points_flip = []
        for hull_v in edge_points:
            edge_points_flip.append(spherical_flipping(flip_radius, hull_v))
        ax.plot([v[0] for v in edge_points_flip], [v[1] for v in edge_points_flip], linestyle="--", linewidth = 1.5, color='purple', zorder = 10)

    return

if __name__ == "__main__":
    flip_radius = 500
        
    laser_scan = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 3.3769965171813965, 3.356032133102417, 3.3727595806121826, 3.38680362701416, 3.3797225952148438, 3.385530471801758, 1.9171745777130127, 1.8068188428878784, 1.7621634006500244, 1.7631354331970215, 1.7454731464385986, 1.7385274171829224, 1.7692523002624512, 1.7680587768554688, 1.7769787311553955, 1.740271806716919, 1.778497338294983, 1.7605037689208984, 1.7465850114822388, 1.7368066310882568, 1.7326892614364624, 1.7430464029312134, 1.7621253728866577, 1.7691127061843872, 1.7830256223678589, 1.7794448137283325, 1.785281777381897, 1.7775700092315674, 11.619831085205078, 11.358251571655273, 11.118313789367676, 10.872776985168457, 10.664834976196289, 10.436408042907715, 10.2370023727417, 10.050822257995605, 9.874287605285645, 9.681632041931152, 9.522553443908691, 9.340362548828125, 9.167712211608887, 9.002094268798828, 8.869200706481934, 8.716163635253906, 8.609018325805664, 8.44872760772705, 8.329237937927246, 8.212705612182617, 8.095394134521484, 7.975321292877197, 7.853057861328125, 7.760853290557861, 7.6427812576293945, 7.564575672149658, 7.45819091796875, 7.3630547523498535, 7.261814117431641, 7.173101425170898, 7.081333637237549, 7.00081205368042, 6.924499034881592, 6.850759029388428, 6.77353048324585, 6.709228038787842, 6.627233028411865, 6.549821376800537, 6.477391242980957, 6.418748378753662, 6.358189582824707, 6.304624557495117, 6.237390518188477, 6.174994468688965, 6.107315540313721, 6.053009510040283, 5.982432842254639, 5.9472222328186035, 5.888853549957275, 5.857597827911377, 5.7949748039245605, 5.731400489807129, 5.709433078765869, 5.642099380493164, 5.617461681365967, 5.571206569671631, 5.526189804077148, 5.497776508331299, 5.453254222869873, 5.419347286224365, 5.354754447937012, 5.330268383026123, 5.281553268432617, 5.249178409576416, 5.220132827758789, 5.190089702606201, 5.1429619789123535, 5.11259651184082, 5.083890438079834, 5.072969436645508, 5.017919540405273, 5.009551525115967, 4.970950126647949, 4.934293746948242, 4.914961814880371, 4.890426158905029, 4.8588056564331055, 4.83143424987793, 4.828697204589844, 4.786685466766357, 4.753010272979736, 4.7253546714782715, 4.7090959548950195, 4.6922502517700195, 4.660085678100586, 4.654407024383545, 4.630891799926758, 4.643182277679443, 4.589930534362793, 4.5783209800720215, 4.566563606262207, 4.535561561584473, 4.527413368225098, 4.525730609893799, 4.500447750091553, 4.478116989135742, 4.463290691375732, 4.465356349945068, 4.432661056518555, 4.425089359283447, 4.421578884124756, 4.395621299743652, 4.387149810791016, 4.3631978034973145, 4.379367828369141, 4.353693008422852, 4.328623294830322, 4.320321559906006, 4.307127475738525, 4.309215545654297, 4.2985310554504395, 4.283123016357422, 4.2740654945373535, 4.282200336456299, 4.257277965545654, 4.256189823150635, 4.232312202453613, 4.228238105773926, 4.241366863250732, 4.223044395446777, 4.23637056350708, 4.217170238494873, 4.208891868591309, 4.214028835296631, 4.209821701049805, 4.201005458831787, 4.202773571014404, 4.1809983253479, 4.181493759155273, 4.164854049682617, 4.188204288482666, 4.168259143829346, 4.177000999450684, 4.169374942779541, 4.165575981140137, 4.17318058013916, 4.187808513641357, 4.166856288909912, 4.183156490325928, 4.1747589111328125, 4.17555570602417, 4.165297031402588, 4.163036346435547, 4.182070255279541, 4.193938255310059, 4.188783168792725, 4.189838886260986, 4.176763534545898, 4.1869354248046875, 4.19609260559082, 4.184962272644043, 4.200784683227539, 4.194479942321777, 4.213934421539307, 4.215202331542969, 4.222422122955322, 4.2405571937561035, 4.223588943481445, 4.237314224243164, 4.2425737380981445, 4.252443313598633, 4.265064239501953, 4.273682117462158, 4.293294906616211, 4.291974067687988, 4.281837463378906, 4.309920787811279, 4.328389644622803, 4.316849231719971, 4.335239410400391, 4.329780578613281, 4.352308750152588, 4.374974250793457, 4.376195430755615, 4.3669281005859375, 4.405158042907715, 4.423472881317139, 4.422791481018066, 4.470305442810059, 4.438962936401367, 4.473112106323242, 4.493249416351318, 4.498071670532227, 4.52437686920166, 4.542567729949951, 4.559791564941406, 4.588547229766846, 4.604719161987305, 4.6173601150512695, 4.625627517700195, 4.657646179199219, 4.6670684814453125, 4.694455146789551, 4.706351280212402, 4.7235493659973145, 4.755340576171875, 4.796244144439697, 4.812355041503906, 4.822240352630615, 4.834167003631592, 4.8784871101379395, 4.9054741859436035, 4.911577224731445, 4.958135604858398, 4.9763689041137695, 5.0085768699646, 5.042031288146973, 5.064081192016602, 5.105923175811768, 5.142247676849365, 5.169546127319336, 5.188181400299072, 5.227230072021484, 5.267539978027344, 5.307473182678223, 5.3357648849487305, 5.37859582901001, 5.402984619140625, 5.44387674331665, 5.487933158874512, 5.54564905166626, 5.583141803741455, 5.627321720123291, 5.680107593536377, 5.725383281707764, 5.783663749694824, 5.824578285217285, 5.876788139343262, 5.921744346618652, 5.971815586090088, 6.014319896697998, 6.075573921203613, 6.1419572830200195, 6.187762260437012, 6.254406452178955, 6.306925296783447, 6.3943328857421875, 6.454428672790527, 6.513002872467041, 6.603212356567383, 6.673946380615234, 6.724701404571533, 6.819855690002441, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 29.783222198486328, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 7.927741527557373, 7.870266437530518, 7.8016510009765625, 7.73752498626709, 7.690275192260742, 7.641604423522949, 7.586978435516357, 7.530237674713135, 7.4621052742004395, 7.4128594398498535, 7.374963760375977, 7.324090003967285, 7.263679027557373, 7.216776371002197, 7.175179958343506, 7.120956897735596, 7.089070796966553, 7.0427117347717285, 7.010378360748291, 6.970056533813477, 6.934691429138184, 6.88340950012207, 6.840488433837891, 6.805203437805176, 6.774470329284668, 6.7359938621521, 6.715569019317627, 6.672825813293457, 6.6395392417907715, 6.6229119300842285, 6.570686340332031, 6.565821170806885, 6.531393527984619, 6.489501953125, 6.468722343444824, 6.422417640686035, 6.4050116539001465, 6.380228042602539, 6.361158847808838, 6.337007522583008, 6.322155475616455, 6.292623996734619, 6.2791876792907715, 6.269611835479736, 6.236152648925781, 6.2149481773376465, 6.1977667808532715, 6.179592132568359, 6.142219543457031, 6.147422790527344, 6.120547771453857, 6.10782527923584, 6.074584484100342, 6.06980037689209, 6.042538642883301, 6.04862117767334, 6.029959678649902, 6.023402214050293, 5.986477375030518, 5.989938259124756, 5.982316017150879, 5.952478885650635, 5.9635090827941895, 5.950704574584961, 5.941098690032959, 5.918259620666504, 5.9135918617248535, 5.904012680053711, 5.892480373382568, 5.896378517150879, 5.897414207458496, 5.872081279754639, 5.861584186553955, 5.864072799682617, 5.848947525024414, 5.844892978668213, 5.856502056121826, 5.851006031036377, 5.827929496765137, 5.854207992553711, 5.838993072509766, 2.935080051422119, 2.931673049926758, 2.761826276779175, 2.786140203475952, 2.7699391841888428, 2.756188154220581, 2.7585530281066895, 2.7603912353515625, 5.810822486877441, 5.8328704833984375, 5.834414958953857, 5.836666584014893, 5.865370273590088, 5.8394904136657715, 5.84691047668457, 5.850865840911865, 5.86622428894043, 5.872786045074463, 5.87168025970459, 5.889029026031494, 5.865400314331055, 5.889889717102051, 5.898161888122559, 5.9004364013671875, 5.916630268096924, 5.931488037109375, 5.9421706199646, 5.932139873504639, 5.953289031982422, 5.992889881134033, 5.981727123260498, 5.985556125640869, 5.995626449584961, 6.021084308624268, 6.026674270629883, 6.046657085418701, 6.064403057098389, 6.053539752960205, 6.085318088531494, 6.109289646148682, 6.1178483963012695, 6.157210350036621, 6.159613132476807, 6.17939567565918, 6.193120956420898, 6.214177131652832, 6.223430633544922, 6.248518466949463, 6.264860153198242, 6.3111162185668945, 6.310152530670166, 6.3459086418151855, 6.3832926750183105, 6.40018892288208, 6.436941146850586, 6.439970970153809, 6.4789652824401855, 6.497858047485352, 6.532124996185303, 6.54265832901001, 6.607362747192383, 6.62883186340332, 6.651089668273926, 6.691003799438477, 6.716070175170898, 6.772172927856445, 6.815639972686768, 6.811223030090332, 6.877920150756836, 6.901495933532715, 6.94259786605835, 6.963021755218506, 7.039237022399902, 7.0824785232543945, 7.105485916137695, 7.153456211090088, 7.198019027709961, 7.244221210479736, 7.267963886260986, 7.360732555389404, 7.394959449768066, 7.431172847747803, 7.5001349449157715, 7.546842575073242, 7.608048439025879, 7.66448450088501, 7.736176013946533, 7.774599552154541, 7.833640098571777, 7.8835062980651855, 7.963433742523193, 8.02428150177002, 8.098939895629883, 8.153071403503418, 8.22989273071289, 8.30982494354248, 8.372066497802734, 8.458617210388184, 8.540531158447266, 8.597468376159668, 8.695690155029297, 8.791470527648926, 8.891853332519531, 8.966327667236328, 9.051070213317871, 9.152573585510254, 9.24516773223877, 9.360851287841797, 9.455897331237793, 9.55361270904541, 9.653809547424316, 9.772342681884766, 9.901865005493164, 10.025065422058105, 10.149003028869629, 10.273436546325684, 10.41304874420166, 10.529757499694824, 10.673654556274414, 10.840950965881348, 10.981267929077148, 11.13786792755127, 11.312654495239258, 11.476556777954102, 11.654081344604492, 11.819765090942383, 12.007779121398926, 12.181966781616211, 12.378203392028809, 12.616496086120605, 12.820670127868652, 13.052766799926758, 13.281794548034668, 13.508487701416016, 13.754913330078125, 14.041008949279785, 14.316238403320312, 14.584033012390137, 14.90377140045166, 15.197644233703613, 15.531282424926758, 15.889352798461914, 16.236608505249023, 16.608139038085938, 17.024993896484375, 17.437545776367188, 3.3768231868743896, 3.362499237060547, 3.3662595748901367, 3.3521058559417725, 3.337287425994873, 3.3295340538024902, 3.467531681060791, 3.6136679649353027, 3.609240770339966, 3.6164565086364746, 3.6272501945495605, 3.616366386413574, 3.6179251670837402, 26.888795852661133, 28.000215530395508, 29.19303321838379, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), 22.704938888549805, 22.94239044189453, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]
    human_list = [[0, -50, -50]]
    robot_pose = [0, 0, 0]

    human_radius = [1 for _ in range(len(human_list))]

    # Input: laser_scan, human_list, robot_pose, human_radius, flip_radius
    # Output: scores for each person in human_list (-1 for invisible)

    scores = human_scoring(laser_scan, human_list, robot_pose, human_radius, flip_radius=flip_radius)

    # Visualization and debugging
    fig, axes = plt.subplots(2, 3)
    for idx, ax in enumerate(axes.flatten()):
        if idx >= len(human_list):
            break
        ax.set_aspect('equal')
        ax.plot([human_list[i][0] for i in range(len(human_list))], [human_list[i][1] for i in range(len(human_list))], 'ro')
        ax.plot(robot_pose[0], robot_pose[1], 'bo')
        ax.set_title(f"Target ID: {str(idx)} | Score: {str(round(scores[idx], 2))}")
        
        for k, human_pose in enumerate(human_list):
            x, y = human_pose[1], human_pose[2]
            if k == idx:
                ax.scatter(human_pose[0], human_pose[1], c='g', s=100, zorder=5)
            else:
                circle = patches.Circle((x, y), human_radius[k], edgecolor='r', facecolor='none', linewidth=2)
                ax.add_patch(circle)

            ax.text(x, y-1.5, str(round(scores[k], 2)), fontsize=12, color="k", ha="center", va="center")  
            ax.text(x, y+1.5, str(k), fontsize=12, color="k", ha="center", va="center")  
        
        visualize_visible_region(laser_scan, human_list, robot_pose, human_radius, flip_radius, target_idx=idx, ax=ax)

    plt.show()
