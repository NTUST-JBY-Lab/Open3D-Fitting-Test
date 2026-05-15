"""
這裡都假設點雲是 2.5D 的
"""
import pyransac3d as pyrsc
import numpy as np
import shapely
import open3d as o3d
import random

from .Util import shapely_poly_to_open3d_mesh, adjustCenterInPlace, clean_crop_aabb, AddBoundaryWeight, rotAlignUpFwd, pointToLineDistance

def fitCreatePlaneRANSAC(points: np.ndarray, silhouette: shapely.Polygon | shapely.MultiPolygon) -> o3d.geometry.TriangleMesh:
    """
    將點雲 fitting 成平面然後將 silhouette 的區塊沿 Y 軸方向投影上去
    """
    # 轉點雲
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # 邊界加厚
    # AddBoundaryWeight(pcd, silhouette)
    # 估計 Normal
    pcd.estimate_normals()
    # RANSAC
    eq_P, _ = pcd.segment_plane(0.01, 3, 1000)

    mesh = shapely_poly_to_open3d_mesh(silhouette)

    # project silhouette
    vert = np.asarray(mesh.vertices)
    vert[:, 1] = -(eq_P[0] * vert[:, 0] + eq_P[2] * vert[:, 2] + eq_P[3]) / eq_P[1]

    return mesh

def fitCreateSphereRANSAC(points: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    將點雲 fitting 成球
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    aabb = pcd.get_axis_aligned_bounding_box()

    center, radius, _ = pyrsc.Sphere().fit(points, thresh=0.01)
    adjustCenterInPlace(pcd, center)

    # 建立以 center 為球心，半徑 radius 的球
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=10)
    vert = np.asarray(mesh.vertices)
    vert[:] = vert[:] + center

    # 切除
    min_bound, max_bound = aabb.get_min_bound(), aabb.get_max_bound()
    avgY = (max_bound[1] + min_bound[1]) / 2
    # 若平均高度 > 圓心的Y -> 留上半
    if avgY > center[1]:
        min_bound[1] = center[1]
        max_bound[1] = np.inf
    # 留下半
    else:
        min_bound[1] = -np.inf
        max_bound[1] = center[1]
    mesh = clean_crop_aabb(mesh, min_bound, max_bound)

    return mesh

def fitCylinderRANSAC(pts: np.ndarray, thresh=0.2, maxIteration=1000):
    """
    使用 RANSAC Fitting 圓柱 
    
    :returns:
    - `center`: Center of the cylinder np.array(1,3) which the cylinder axis is passing through.
    - `axis`: Vector describing cylinder's axis np.array(1,3).
    - `radius`: Radius of cylinder.
    - `inliers`: Inlier's index from the original point cloud.
    ---
    """
    n_points = pts.shape[0]
    best_inliers = np.array([])
    best_line_pt = None
    best_axis = None
    best_radius = None

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.estimate_normals()

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    for it in range(maxIteration):

        # Samples 2 random points
        id_samples = random.sample(range(0, n_points), 2)
        p1, p2 = pts[id_samples]
        n1, n2 = normals[id_samples]

        #################################################################################
        # Compute Model
        # Ref: https://github.com/PointCloudLibrary/pcl/blob/master/sample_consensus/include/pcl/sample_consensus/impl/sac_model_cylinder.hpp
        #################################################################################
        w = n1 + p1 - p2
        line_dir = np.cross(n1, n2)

        b = np.dot(n1, n2)
        c = np.dot(n2, n2)
        d = np.dot(n1, w)
        e = np.dot(n2, w)
        denominator = np.dot(line_dir, line_dir) # Squared Norm

        # Compute the line parameters of the two closest points
        if denominator < 1e-8:     # The lines are almost parallel
            sc = 0
        else:
            sc = (b * e - c * d) / denominator

        # point_on_axis, axis_direction
        line_pt = p1 + n1 + sc * n1
        line_dir /= np.linalg.norm(line_dir) # Normalized

        # radius
        radius = 0.5 * (pointToLineDistance(p1, line_pt, line_dir) + pointToLineDistance(p2, line_pt, line_dir))

        #################################################################################
        # Find Inlier
        #################################################################################
        # Distance from a point to a line
        pt_id_inliers = []  # list of inliers ids
        vecC_stakado = np.repeat([line_dir], n_points, axis=0)
        dist_pt = np.cross(vecC_stakado, (line_pt - pts))
        dist_pt = np.linalg.norm(dist_pt, axis=1)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = np.where(np.abs(dist_pt - radius) <= thresh)[0]

        if len(pt_id_inliers) > len(best_inliers):
            best_inliers = pt_id_inliers
            best_line_pt= line_pt
            best_axis = line_dir
            best_radius = radius

    if len(best_inliers) > 0:
        # Project Point Cloud's center onto cylinder's axis
        pcd_center = pcd.get_center()
        projected_center = best_line_pt + np.dot(pcd_center - best_line_pt, best_axis) * best_axis
    else:
        projected_center = None

    return projected_center, best_axis, best_radius, best_inliers


def createCylinder(points: np.ndarray, center: np.ndarray, axis: np.ndarray, radius: float) -> o3d.geometry.TriangleMesh:
    """ 建立圓柱 """
    # Step 1. 找高、寬 #####################################################################################################
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    
    # 我們把點雲的 axis 軸轉到 (0, 0, 1)，且保持 Y 軸方向朝上
    # 旋轉後，Z 方向的長為圓柱的高，X 方向的長為圓柱的寬
    rot = rotAlignUpFwd(np.array([0, 1, 0]), axis) # size: 3 x 3
    pts = np.asarray(pcd.points) # size: N x 3

    # for each i:
    #   pts[i] = (rot @ pts[i].T).T = pts[i] @ rot.T
    pts -= center
    pts[:] = pts @ rot.T

    # 找圓柱的範圍
    # Z 方向的長為圓柱的高，X 方向的長為圓柱的寬
    min_bound = pcd.get_axis_aligned_bounding_box().get_min_bound()
    max_bound = pcd.get_axis_aligned_bounding_box().get_max_bound()
    height = max_bound[2] - min_bound[2]
    
    # Y 方向 average > 0 -> 只取上半；否則，只取下半圓柱
    if (max_bound[1] + min_bound[1]) / 2 > 0:
        min_bound[1] = 0
        max_bound[1] = np.inf
    else:
        min_bound[1] = -np.inf
        max_bound[1] = 0

    # Step 2. 建立圓柱 ####################################################################################################
    # 預設：中心 (0, 0, 0)、軸向 (0, 0, 1)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius, height + 1) # height 多取一些，等下再切掉

    # 對 cylinder 裁切
    cylinder = clean_crop_aabb(cylinder, min_bound, max_bound)
    
    # 旋轉使 (0, 0, 1) 變 axis + 平移使 (0, 0, 0) 變 center
    # rot = np.linalg.inv(rotAlignUpFwd(np.array([0, 1, 0]), axis))
    rot = np.linalg.inv(rot)
    pts = np.asarray(cylinder.vertices)
    pts[:] = pts @ rot.T + center

    return cylinder

