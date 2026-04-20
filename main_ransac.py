"""
給定一個資料夾

對資料夾裡面的每個 obj 檔，做 Sample -> 2D Alphashape -> RANSAC -> Export
"""
import open3d as o3d
import os
import time
import alphashape
import numpy as np
import shapely
from Samples import sample_mesh_with_raycast, sample_along_edges
from Util import shapely_poly_to_open3d_mesh
import pyransac3d as pyrsc

INPUT_DIR = "main_ransac"

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
                            , np.asarray(sample_along_edges(mesh, 0.01).points)])

        if points.size == 0:
            continue

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(os.path.join(INPUT_DIR, f"{obj_name}_pcd.ply"), pcd)
        print("Sample:", time.perf_counter() - s)

        # Step2. Alpha Shape #################################
        s = time.perf_counter()
        try:
            points_2d = [(p[0], p[2]) for p in pcd.points]
            result = alphashape.alphashape(points_2d, 50)
            result = result.simplify(0.01)
            mesh = shapely_poly_to_open3d_mesh(result)
            o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_alphashape.obj"), mesh)
        except:
            continue
        print("Alpha Shape:", time.perf_counter() - s)

        # Step3. RANSAC ####################################
        s = time.perf_counter()
        eq_P, inliers_P = pyrsc.Plane().fit(np.asarray(pcd.points))
        center, radius, inliers_S = pyrsc.Sphere().fit(np.asarray(pcd.points))
        original_size = np.max(mesh.get_axis_aligned_bounding_box().get_extent()[[0, 2]])

        # 如果更貼近平面 or Fitting 出的球太大了 -> 用平面 fitting
        if inliers_P.size >= inliers_S.size or radius > 2 * original_size:
            # project alphashape
            vert = np.asarray(mesh.vertices)
            for i in range(len(vert)):
                vert[i, 1] = -(eq_P[0] * vert[i, 0] + eq_P[2] * vert[i, 2] + eq_P[3]) / eq_P[1]
        else:
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
            vert = np.asarray(mesh.vertices)
            for i in range(len(vert)):
                vert[i] = vert[i] + center

        o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_RANSAC.obj"), mesh)
        print("RANSAC:", time.perf_counter() - s)
        print("")


if __name__ == "__main__":
    for file in os.listdir(INPUT_DIR):
        if file.endswith('_alphashape.obj') or file.endswith('_pcd.ply') or file.endswith('_RANSAC.obj'):
            os.remove(os.path.join(INPUT_DIR, file))

    main()
