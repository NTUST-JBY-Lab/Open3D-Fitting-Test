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
from open3d_fitting_test.Util import shapely_poly_to_open3d_mesh, alaphashape_union2D, clean_crop_aabb, adjustCenterInPlace, AddBoundaryWeight, almostGreaterAlongAxis
from open3d_fitting_test.Fit import createCylinder, fitCylinderRANSAC
from Result import cleanup_result
import math
import pyransac3d as pyrsc
import argparse
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError

INPUT_DIR = os.path.join(os.path.dirname(__file__), "ransac_test")

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Input directory")
parser.add_argument("-t", help="Ransac Iteration")
parser.add_argument("-T", help="Ransac Threshold")
parser.add_argument("-r", help="Relative tolerance to prefer plane")
parser.add_argument("-R", help="Construction resolution of sphere and cylinder")
args = parser.parse_args()

if args.i is not None:
    INPUT_DIR = args.i

ITERATION = int(args.t) if args.t is not None else 1000
THRESHOLD = float(args.T) if args.T is not None else 0.01
REL_TOL = float(args.r) if args.r is not None else 0.2
RESOLUTION = int(args.R) if args.R is not None else 10

log = open('log.txt', 'w')

def extendTooSmall(mesh: o3d.geometry.TriangleMesh, thresh = 0.01):
    """ 如果 mesh 投影到 XZ 平面的 Minimum bounding box 的長或寬 < thresh，則代表太小 """
    skip = False

    pts_3d = np.asarray(mesh.vertices)
    pts_2d = pts_3d[:, [0, 2]]

    try:
        envelope: shapely.Polygon = shapely.minimum_rotated_rectangle(shapely.MultiPoint(
            pts_2d[ConvexHull(pts_2d).vertices] # 只拿 convex hull 的邊界點算 rotated bounding box
        ))
        
        if not isinstance(envelope, shapely.Polygon):
            skip = True
        else:
            w = math.dist(envelope.exterior.coords[0], envelope.exterior.coords[1])
            h = math.dist(envelope.exterior.coords[1], envelope.exterior.coords[2])
            if w < thresh or h < thresh:
                skip = True
    except QhullError:
        skip = True

    return skip

def determineSampleStep(mesh: o3d.geometry.TriangleMesh) -> float:
    dx, dz = mesh.get_axis_aligned_bounding_box().get_extent()[[0, 2]]

    return max(max(dx, dz) / 100, 0.01)

def main():
    for file in os.listdir(INPUT_DIR):
    # for file in ["14.obj"]:
        if not file.endswith(".obj"):
            continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(INPUT_DIR, file))
        obj_name = file.removesuffix(".obj")
        print("== Processing: ", obj_name)

        # Step1.建立點雲 ####################################
        # 如果太小 -> 忽略
        s = time.perf_counter()
        if extendTooSmall(mesh):
            print("Too small -> Skip\n")
            continue

        sample_step = determineSampleStep(mesh)
        # 同時用 Raycast 和 Edge Sample
        points, normals = sample_mesh_with_raycast(mesh, sample_step)
        print("\tSample Raycast:", time.perf_counter() - s, "s"); sR = time.perf_counter()
        points = np.vstack([points
                            , sample_along_edges(mesh, sample_step)])
        print("\tSample Edges:", time.perf_counter() - sR, "s")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        del points
        points_2d = np.asarray(pcd.points)[:, [0, 2]]
        
        print("Sample:", time.perf_counter() - s, "s")
        o3d.io.write_point_cloud(os.path.join(INPUT_DIR, f"{obj_name}_pcd.ply"), pcd)

        # Step2. Alpha Shape #################################
        s = time.perf_counter()
        try:
            alphashape = shapely.MultiPolygon(alaphashape_union2D(points_2d, alpha=1 / (2 * sample_step))).simplify(0.005).buffer(0)
            mesh = shapely_poly_to_open3d_mesh(alphashape)
            o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_alphashape.obj"), mesh)
        except Exception as e:
            print(e)
            print("")
            continue

        print("Alpha Shape:", time.perf_counter() - s, "s")

        # Step3. RANSAC ####################################
        s = time.perf_counter()
        pcd.estimate_normals()
        # 一次 fit 三個
        sR = time.perf_counter(); eq_P, inliers_P = pcd.segment_plane(THRESHOLD, 3, ITERATION);                                                                print(f"\tRANSAC Plane: {time.perf_counter() - sR} s")
        sR = time.perf_counter(); center_S, radius_S, inliers_S = pyrsc.Sphere().fit(np.asarray(pcd.points), maxIteration=ITERATION, thresh=THRESHOLD);        print(f"\tRANSAC Sphere: {time.perf_counter() - sR} s")
        sR = time.perf_counter(); center_C, axis_C, radius_C, inliers_C = fitCylinderRANSAC(np.asarray(pcd.points), maxIteration=ITERATION, thresh=THRESHOLD); print(f"\tRANSAC Cylinder: {time.perf_counter() - sR} s")

        aabb = pcd.get_axis_aligned_bounding_box()
        original_size = np.max(aabb.get_extent()[[0, 2]])

        # Log
        log.write(f"{obj_name} (#pts: {np.asarray(pcd.points).shape[0]}, max_width: {original_size}, sample_step: {sample_step})\n")
        log.write(f"Plane: {eq_P} ({len(inliers_P)})\nSphere: {center_S}, {radius_S} ({inliers_S.shape[0]})\nCylinder: center={center_C}, axis={axis_C}, radius={radius_C} ({inliers_C.shape[0]})\n")

        # 看哪個比較接近就用哪個 ##########################################################
        choice = sorted([('Plane', len(inliers_P)), ('Sphere', inliers_S.size), ('Cylinder', inliers_C.size)]
                        , key=lambda x: x[1], reverse=True)
        fit_target = None

        for i in range(len(choice)):
            match choice[i][0]:
                case 'Plane':
                    fit_target = 'Plane'
                    break

                case 'Sphere':
                    # Fitting 出的球太大了
                    if radius_S > 2 * original_size:
                        log.write("Sphere too big -> Next Choice\n")
                        continue
                    # 兩者很相近 -> 傾向用平面
                    elif math.isclose(inliers_S.size, len(inliers_P), rel_tol=REL_TOL):
                        log.write("Sphere and Plane are almost same -> Prefer Plane\n")
                        continue

                    fit_target = 'Sphere'
                    break
                
                case 'Cylinder':
                    # Fitting 出的圓柱太大了
                    if radius_C > 2 * original_size:
                        log.write("Cylinder too big -> Next Choice\n")
                        continue
                    # 兩者很相近 -> 傾向用平面
                    elif math.isclose(inliers_C.size, len(inliers_P), rel_tol=REL_TOL):
                        log.write("Cylinder and Plane are almost same -> Prefer Plane\n")
                        continue

                    fit_target = 'Cylinder'
                    break
        
        if fit_target is None:
            fit_target = 'Plane'
        
        # Create Result ############################################################
        match fit_target:
            case 'Plane':
                log.write("Fit Plane\n")
                # project alphashape
                vert = np.asarray(mesh.vertices)
                vert[:, 1] = -(eq_P[0] * vert[:, 0] + eq_P[2] * vert[:, 2] + eq_P[3]) / eq_P[1]
                mesh.paint_uniform_color(np.array([0, 0, 1]))

            case 'Sphere':
                log.write("Fit Sphere\n")
                adjustCenterInPlace(pcd, center_S)

                # 建立以 center 為球心，半徑 radius 的球
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius_S, resolution=RESOLUTION)
                vert = np.asarray(mesh.vertices)
                vert[:] = vert[:] + center_S

                # 切除
                min_bound, max_bound = aabb.get_min_bound(), aabb.get_max_bound()
                # 若幾乎都在圓心之上 -> 留上半
                if almostGreaterAlongAxis(pcd, 'y', center_S[1]):
                    min_bound[1] = center_S[1]
                    max_bound[1] = np.inf
                # 留下半
                else:
                    min_bound[1] = -np.inf
                    max_bound[1] = center_S[1]
                mesh = clean_crop_aabb(mesh, min_bound, max_bound)
                mesh.paint_uniform_color(np.array([0, 1, 0]))

            case 'Cylinder':
                log.write("Fit Cylinder\n")
                mesh = createCylinder(np.asarray(pcd.points), center_C, axis_C, radius_C, RESOLUTION)
                mesh.paint_uniform_color(np.array([1, 0, 0]))

        log.write("\n")
        log.flush()
        o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_RANSAC.obj"), mesh)
        print("RANSAC:", time.perf_counter() - s, "s")
        print("")

if __name__ == "__main__":
    cleanup_result(INPUT_DIR)
    main_start = time.perf_counter()
    main()
    print(f"\nTotal Time: {time.perf_counter() - main_start} s")
