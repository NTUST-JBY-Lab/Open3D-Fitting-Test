import shapely
import numpy as np
import open3d as o3d
import os
import alphashape

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

def cleanup_result(directory: str, file_extension: list[str] = ['_alphashape.obj', '_pcd.ply', '_RANSAC.obj', '_poisson.obj', '_silhouette.obj']):
    """
    對於 directory 下的所有檔案，如果檔名以 file_extension 中的其中一個值結尾，則刪掉
    """
    for file in os.listdir(directory):
        # 如果 file 以任何一個 file_extension 結尾
        if any(file.endswith(ext) for ext in file_extension):
            # delete
            os.remove(os.path.join(directory, file))

def alaphashape_union(points_2d: list[tuple[float, float]], *, alpha: float = 50) -> list[shapely.Polygon]:
    """
    三角化 -> 留外切圓半徑夠小的 -> Union -> 留 exterior -> simplify
    """
    if alpha == 0:
        return [shapely.MultiPoint(points_2d).convex_hull]

    faces = []
    # 對點雲做 Delaunay 然後計算每個面的外接圓半徑
    for simplex, radius in alphashape.alphasimplices(points_2d):
        if radius < 1 / alpha:
            faces.append(simplex.tolist())

    # 將所有留下的面做 Union
    Union = shapely.unary_union([shapely.Polygon([points_2d[vertex] for vertex in F]) for F in faces])

    return [shapely.Polygon(P.exterior).simplify(0.01) for P in shapely.get_parts(Union) if isinstance(P, shapely.Polygon)]
