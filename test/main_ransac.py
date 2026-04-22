"""
給定一個資料夾

對資料夾裡面的每個 obj 檔，做 Sample -> 2D Alphashape -> RANSAC -> Export
"""
import open3d as o3d
import os
import time
import numpy as np
import shapely
from open3d_fitting_test.Samples import sample_mesh_with_raycast, sample_along_edges
from open3d_fitting_test.Util import shapely_poly_to_open3d_mesh, alaphashape_union2D, clean_crop_aabb, adjustCenterInPlace
from Result import cleanup_result
import math
import pyransac3d as pyrsc
from typing import Literal
import argparse

INPUT_DIR = "ransac_test"

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Input Directory")
args = parser.parse_args()

if args.i is not None:
    INPUT_DIR = args.i

def main():
    for file in os.listdir(INPUT_DIR):
    # for file in ["14.obj"]:
        if not file.endswith(".obj"):
            continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(INPUT_DIR, file))
        obj_name = file.removesuffix(".obj")

        # Step1.建立點雲 ####################################
        s = time.perf_counter()
        points, normals = sample_mesh_with_raycast(mesh, 0.01)
        points = np.vstack([points
                            , sample_along_edges(mesh, 0.01)])
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        points_2d = [(p[0], p[2]) for p in pcd.points]

        # 如果太小 -> 忽略
        envelope: shapely.Polygon = shapely.minimum_rotated_rectangle(shapely.MultiPoint(points_2d)).normalize()
        w = math.dist(envelope.exterior.coords[0], envelope.exterior.coords[1])
        h = math.dist(envelope.exterior.coords[1], envelope.exterior.coords[2])
        if w < 0.01 or h < 0.01:
            continue
        
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(os.path.join(INPUT_DIR, f"{obj_name}_pcd.ply"), pcd)
        print("Sample:", time.perf_counter() - s)

        # Step2. Alpha Shape #################################
        s = time.perf_counter()
        try:
            result = alaphashape_union2D(points_2d, alpha=50)
            mesh = shapely_poly_to_open3d_mesh(max(result, key=lambda p: p.area).simplify(0.01))
            o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_alphashape.obj"), mesh)
        except Exception as e:
            print(e)
            continue
        print("Alpha Shape:", time.perf_counter() - s)

        # Step3. RANSAC ####################################
        s = time.perf_counter()
        eq_P, inliers_P = pyrsc.Plane().fit(np.asarray(pcd.points))
        center, radius, inliers_S = pyrsc.Sphere().fit(np.asarray(pcd.points))

        aabb = pcd.get_axis_aligned_bounding_box()
        original_size = np.max(aabb.get_extent()[[0, 2]])

        # 如果更貼近平面 or Fitting 出的球太大了 -> 用平面 fitting or fitting 的球球心太高
        if inliers_P.size >= inliers_S.size or radius > 2 * original_size or center[1] >= aabb.get_min_bound()[1]:
            # project alphashape
            vert = np.asarray(mesh.vertices)
            for i in range(len(vert)):
                vert[i, 1] = -(eq_P[0] * vert[i, 0] + eq_P[2] * vert[i, 2] + eq_P[3]) / eq_P[1]
        else:
            adjustCenterInPlace(pcd, center)

            # 建立以 center 為球心，半徑 radius 的球
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=10)
            vert = np.asarray(mesh.vertices)
            for i in range(len(vert)):
                vert[i] = vert[i] + center

            # 切除
            max_bound = aabb.get_max_bound()
            max_bound[1] = np.inf # 高度不切最高
            mesh = clean_crop_aabb(mesh, aabb.get_min_bound(), max_bound)

        o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_RANSAC.obj"), mesh)
        print("RANSAC:", time.perf_counter() - s)
        print("")

if __name__ == "__main__":
    cleanup_result(INPUT_DIR)
    main()
