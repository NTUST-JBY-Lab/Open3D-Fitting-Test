"""
給定一個資料夾

對資料夾裡面的每個 obj 檔，做 Sample -> 2D Alphashape -> Poisson -> Export
"""
import open3d as o3d
import os
import time
import alphashape
import numpy as np
import shapely
from Samples import sample_mesh_with_raycast
from Util import shapely_poly_to_open3d_mesh

INPUT_DIR = "main_poisson"
POISSON_DEPTH = 6
POISSON_SCALE = 1.5

def main():
    for file in os.listdir(INPUT_DIR):
        if not file.endswith(".obj"):
            continue

        mesh = o3d.io.read_triangle_mesh(os.path.join(INPUT_DIR, file))
        obj_name = file.removesuffix(".obj")

        # Step1.建立點雲 ####################################
        s = time.perf_counter()
        points, normals = sample_mesh_with_raycast(mesh, 0.01)

        if points.size == 0:
            continue

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        pcd.normals = o3d.utility.Vector3dVector(normals)
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

        # Step3. Poisson ####################################
        s = time.perf_counter()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=POISSON_DEPTH, scale=POISSON_SCALE)
        o3d.io.write_triangle_mesh(os.path.join(INPUT_DIR, f"{obj_name}_poisson.obj"), mesh)
        print("Poisson:", time.perf_counter() - s)


if __name__ == "__main__":
    main()
