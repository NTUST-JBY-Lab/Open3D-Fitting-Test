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
from open3d_fitting_test.Util import shapely_poly_to_open3d_mesh, alaphashape_union2D, clean_crop_aabb, adjustCenterInPlace, AddBoundaryWeight
from Result import cleanup_result
import math
import pyransac3d as pyrsc
import argparse

INPUT_DIR = os.path.join(os.path.dirname(__file__), "ransac_test")

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Input Directory")
args = parser.parse_args()

if args.i is not None:
    INPUT_DIR = args.i

log = open('log.txt', 'w')

def main():
    for file in os.listdir(INPUT_DIR):
    # for file in ["14.obj"]:
        if not file.endswith(".obj"):
            continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(INPUT_DIR, file))
        obj_name = file.removesuffix(".obj")
        print("== Processing: ", obj_name)

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
        print("Sample:", time.perf_counter() - s)

        # Step2. Alpha Shape #################################
        s = time.perf_counter()
        try:
            alphashape = max(alaphashape_union2D(points_2d, alpha=50), key=lambda p: p.area)
            mesh = shapely_poly_to_open3d_mesh(alphashape.simplify(0.01))
            o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_alphashape.obj"), mesh)
        except Exception as e:
            print(e)
            continue

        # AddBoundaryWeight(pcd, alphashape)
        o3d.io.write_point_cloud(os.path.join(INPUT_DIR, f"{obj_name}_pcd.ply"), pcd)

        print("Alpha Shape:", time.perf_counter() - s)

        # Step3. RANSAC ####################################
        s = time.perf_counter()
        pcd.estimate_normals()
        eq_P, inliers_P = pcd.segment_plane(0.01, 3, 1000)
        center, radius, inliers_S = pyrsc.Sphere().fit(np.asarray(pcd.points))
        log.write(f"{obj_name}\nPlane: {eq_P} ({len(inliers_P)})\nSphere: {center}, {radius} ({inliers_S.size})\n")

        aabb = pcd.get_axis_aligned_bounding_box()
        original_size = np.max(aabb.get_extent()[[0, 2]])

        # 看哪個比較接近就用哪個
        fit_plane = len(inliers_P) >= inliers_S.size

        # Fitting 出的球太大了 or fitting 的球球心太高 -> 用平面 fitting
        if radius > 2 * original_size:# or center[1] >= aabb.get_min_bound()[1]:
            log.write("Sphere too big -> Force Plane\n")
            fit_plane = True
        # 兩者很相近 -> 傾向用球
        elif math.isclose(inliers_S.size, len(inliers_P), rel_tol=0.1):
            log.write("Spher and Plane are almost same -> Prefer Plane\n")
            fit_plane = True

        if  fit_plane:
            log.write("Fit Plane\n")
            # project alphashape
            vert = np.asarray(mesh.vertices)
            vert[:, 1] = -(eq_P[0] * vert[:, 0] + eq_P[2] * vert[:, 2] + eq_P[3]) / eq_P[1]
        else:
            log.write("Fit Sphere\n")
            adjustCenterInPlace(pcd, center)

            # 建立以 center 為球心，半徑 radius 的球
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=10)
            vert = np.asarray(mesh.vertices)
            vert[:] = vert[:] + center

            # 切除
            max_bound = aabb.get_max_bound()
            max_bound[1] = np.inf # 高度不切最高
            mesh = clean_crop_aabb(mesh, aabb.get_min_bound(), max_bound)

        log.flush()
        o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_RANSAC.obj"), mesh)
        print("RANSAC:", time.perf_counter() - s)
        print("")

if __name__ == "__main__":
    cleanup_result(INPUT_DIR)
    main()
