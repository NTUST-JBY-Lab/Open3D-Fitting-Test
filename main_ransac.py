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


if __name__ == "__main__":
    for file in os.listdir(INPUT_DIR):
        if file.endswith('_alphashape.obj') or file.endswith('_pcd.ply') or file.endswith('_RANSAC.obj'):
            os.remove(os.path.join(INPUT_DIR, file))

    main()
