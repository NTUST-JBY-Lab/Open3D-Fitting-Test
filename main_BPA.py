"""
測試用 BPA 或 Alphashape 找輪廓
"""
import open3d as o3d
import time
import numpy as np
from open3d_fitting_test.Samples import sample_mesh_with_raycast, sample_along_edges
from open3d_fitting_test.Util import shapely_poly_to_open3d_mesh
import alphashape
from shapely.plotting import *
import matplotlib.pyplot as plt
import shapely

BPA = False

INPUT = "input3"
mesh = o3d.io.read_triangle_mesh(f"{INPUT}.obj")

# project to ground
for vert in mesh.vertices:
    vert[1] = 0

# s = time.perf_counter()
# # 沿著邊取樣 ###########################################
# pcd = sample_along_edges(mesh, 0.01)
# print(time.perf_counter() - s)

# 找屋頂時的取樣 ########################################
s = time.perf_counter()
points, _ = sample_mesh_with_raycast(mesh, 0.01)
print("Sample:", time.perf_counter() - s)

# pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([pcd.points, points])))
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd.remove_duplicated_points()

if BPA:
    # BPA
    s = time.perf_counter()
    pcd.normals = o3d.utility.Vector3dVector([(0, 1, 0) for _ in range(len(pcd.points))])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.02]))
    print("BPA:", time.perf_counter() - s)
else:
    # alphashape
    s = time.perf_counter()
    points_2d = [(p[0], p[2]) for p in pcd.points]
    result = alphashape.alphashape(points_2d, 50)
    result = result.simplify(0.01)
    print("Alpha Shape:", time.perf_counter() - s)

    plot_polygon(result)
    plt.show()
    mesh = shapely_poly_to_open3d_mesh(result)

o3d.io.write_triangle_mesh(f"{INPUT}_silhouette.obj", mesh)
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True, mesh_show_wireframe=True)

