"""
用來測試 Poisson
"""
import open3d as o3d
import time
from open3d_fitting_test.Samples import sample_mesh_with_raycast

INPUT = "input3"
mesh = o3d.io.read_triangle_mesh(f"{INPUT}.obj")

# 找屋頂時的取樣 ########################################
s = time.perf_counter()
points, normals = sample_mesh_with_raycast(mesh, 0.01)
print("Sample:", time.perf_counter() - s)

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
pcd.normals = o3d.utility.Vector3dVector(normals)

# Poisson
s = time.perf_counter()
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=4, scale=4)
print("Poisson:", time.perf_counter() - s)

mesh.compute_triangle_normals()
o3d.io.write_triangle_mesh(f"{INPUT}_poisson.obj", mesh)
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
