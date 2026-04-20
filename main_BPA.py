"""
測試用 BPA 或 Alphashape 找輪廓
"""
import open3d as o3d
import time
import numpy as np
from Samples import sample_mesh_with_raycast, sample_along_edges
import alphashape
from shapely.plotting import *
import matplotlib.pyplot as plt
import shapely

def shapely_poly_to_open3d_mesh(poly: shapely.Polygon, z_value=0.0):
    """
    將 Shapely Polygon 轉換為 Open3D TriangleMesh
    :param poly: shapely.geometry.Polygon 物件
    :param z_value: 投影到 3D 空間時的 Y 座標 (或是 Z 座標，依你的坐標系而定)
    :return: o3d.geometry.TriangleMesh
    """
    if poly.is_empty:
        return o3d.geometry.TriangleMesh()

    # 1. 對多邊形進行三角剖分
    # triangulate 會回傳一組三角形列表，我們過濾掉不在多邊形內部的三角形 (處理凹角與孔洞)
    tris = [t for t in shapely.get_parts(shapely.constrained_delaunay_triangles(poly)) if isinstance(t, shapely.Polygon)]
    
    all_vertices = []
    all_triangles = []
    vert_map = {}

    for tri in tris:
        current_tri_indices = []
        # tri.exterior.coords 包含 4 個點 (起點與終點重複)，取前 3 個
        for coord in list(tri.exterior.coords)[:3]:
            # 使用 coordinate 作為 key 來避免重複頂點
            pt = (coord[0], z_value, coord[1]) # 假設你之前是投影到 XZ 平面
            
            if pt not in vert_map:
                vert_map[pt] = len(all_vertices)
                all_vertices.append(pt)
            
            current_tri_indices.append(vert_map[pt])
        
        all_triangles.append(current_tri_indices)

    # 2. 建立 Open3D Mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
    
    # 計算法向量以利顯示
    mesh.compute_vertex_normals()
    
    return mesh

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

