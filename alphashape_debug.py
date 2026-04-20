"""
alphashape.alphashape 在找輪廓時可能會變成一條線，應該是 Polygonize 那段有問題
"""
import open3d as o3d
from scipy.spatial import Delaunay
import numpy as np
import alphashape
import shapely
import pandas as pd

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

def alphashape1():
    pcd = o3d.io.read_point_cloud("alphashape_debug/0_pcd.ply")

    # 1. 做三角化
    tris = Delaunay([(x, z) for x, y, z in np.asarray(pcd.points)])

    # 2. 保留外切圓半徑小於 1/50 的三角形
    radius_list = []
    count = 0
    faces = []
    for simplex in tris.simplices:
        try:
            radius_list.append(alphashape.circumradius(tris.points[simplex]))
            count += 1

            # radius 夠小 -> 留下
            if radius_list[-1] < 1 / 50:
                faces.append(simplex.tolist())
        except:
            pass

    print(len(faces))
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((x, 0, z) for x, z in tris.points)
                                  , o3d.utility.Vector3iVector(faces))
        ], mesh_show_wireframe=True, mesh_show_back_face=True)



def alphashape2():
    pcd = o3d.io.read_point_cloud("alphashape_debug/0_pcd.ply")
    points_2d = [(x, z) for x, y, z in np.asarray(pcd.points)]

    faces = []
    for simplex, radius in alphashape.alphasimplices(points_2d):
        if radius < 1 / 50:
            faces.append(simplex.tolist())

    print(len(faces))
    o3d.visualization.draw_geometries([
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((x, 0, z) for x, z in points_2d)
                                  , o3d.utility.Vector3iVector(faces))
        ], mesh_show_wireframe=True, mesh_show_back_face=True)
    
def alphashape3():
    pcd = o3d.io.read_point_cloud("alphashape_debug/0_pcd.ply")
    points_2d = [(x, z) for x, y, z in np.asarray(pcd.points)]
    
    shapes = alphashape.alphashape(points_2d, 50)

    o3d.visualization.draw_geometries([shapely_poly_to_open3d_mesh(shapes)], mesh_show_wireframe=True, mesh_show_back_face=True)

if __name__ == "__main__":
    alphashape1()
    alphashape2()
    alphashape3()