import shapely
from shapely.strtree import STRtree
import numpy as np
import numba as nb
import open3d as o3d
import alphashape
from typing import Literal
from scipy.spatial.transform import Rotation

def shapely_poly_to_open3d_mesh(poly: shapely.Polygon | shapely.MultiPolygon, y_value=0.0):
    """
    將 Shapely Polygon 轉換為 Open3D TriangleMesh。
    - 如果 Polygon 是2維的，將 Polygon 的頂點從 (x, y) -> (x, y_value, y)
    - 如果 Polygon 是3維的，將 Polygon 的頂點從 (x, y, z) -> (x, z, y)

    :param poly: shapely.geometry.Polygon 物件
    :param y_value: 投影到 3D 空間時的 Y 座標
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
            if len(coord) == 2:
                pt = (coord[0], y_value, coord[1]) # 假設你之前是投影到 XZ 平面
            else:
                pt = (coord[0], coord[2], coord[1])
            
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

def rotFromToVec(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """ 建立一個旋轉矩陣，將 vec1 旋轉為 vec2 """
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    # Rotation Axis
    axis = np.cross(vec1, vec2)

    # Rotation Angle (in radiance)
    rad = np.arccos(np.dot(vec1, vec2))

    rot = Rotation.from_rotvec(axis * rad)

    return rot.as_matrix()

def rotAlignUpFwd(up: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """ 建立一個旋轉，將 fwd 轉到 (0, 0, 1) 的方向，將 fwd X up 轉到 (1, 0, 0)，將 (fwd X up) X fwd 轉到 (0, 1, 0) """
    up = up / np.linalg.norm(up)
    fwd = fwd / np.linalg.norm(fwd)

    # 向右
    R = np.cross(fwd, up)

    # 向上
    U = np.cross(R, fwd)

    return np.array([
        R,
        U,
        fwd
    ])

def pointToLineDistance(p0: np.ndarray, line_pt: np.ndarray, line_dir: np.ndarray) -> float:
    """ 計算 p0 到直線的距離，直線過 line_pt、方向為 line_dir """
    # Calculate the distance from the point to the line
    # Line: P1 ~ P2
    # D = ||(P2-P1) x (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p1-p0)) / norm(p2-p1)

    return np.linalg.norm(np.cross(line_dir, line_pt - p0)) / np.linalg.norm(line_dir)

############################################################################################################################
# Alpha Shape
############################################################################################################################
@nb.guvectorize([(nb.float64[:, :], nb.float64[:])], '(m, n) -> ()', cache=True)
def circumradius(points: np.ndarray, res: np.ndarray):
    """
    傳入一個 M * N 的點陣列，代表有 M 個 N 維的點，回傳這些點的外接圓半徑
    - Reference: alphashape.circumradius
    """
    rows, _ = points.shape
    A = np.zeros((rows + 1, rows + 1), dtype=points.dtype)
    A[:rows, :rows] = 2 * np.dot(points, points.T)
    A[:rows, -1] = np.ones(rows)
    A[-1, :rows] = np.ones(rows)

    b = np.hstack((
        np.sum(points * points, axis=1),
        np.ones((1))
    ))

    if np.linalg.det(A) == 0:
        res[0] = np.nan
    else:
        circumcenter = np.linalg.solve(A, b)[:-1]
        res[0] = np.linalg.norm(points[0, :] - np.dot(circumcenter, points))

def alaphashape_union2D(points_2d: list[tuple[float, float]], *, alpha: float = 50) -> list[shapely.Polygon]:
    """
    Delaunay 三角化 -> 留外切圓半徑夠小的 -> Union
    """
    if alpha == 0:
        return [shapely.MultiPoint(points_2d).convex_hull]
    
    from scipy.spatial import Delaunay
    import time
    
    print("== alphashape union 2D ==")

    s = time.perf_counter()
    # 1. 做三角化
    tris = Delaunay(points_2d)
    print("\tDelaunay: ", time.perf_counter() - s, "s")

    s = time.perf_counter()
    # 2. 保留外切圓半徑小於 1/alpha 的三角形
    # tris.simplicies is (N, 3) (dtype=int) -> N 個 simplex，每個 simplex 由 3 個 2D 點儲存，每列為點的 index
    # tris.points is (M, 2) (dtype=float64) -> M 個 2D 點，每列為點的 xy 座標
    simplices_coord = tris.points[tris.simplices] # (N, 3, 2)

    radius = circumradius(simplices_coord) # (N, 1)

    faces = tris.simplices[radius < 1 / alpha]

    print("\tCircumradius: ", time.perf_counter() - s, "s")

    s = time.perf_counter()
    # 3. 將所有留下的面做 Union
    Union = shapely.unary_union([
        shapely.Polygon([points_2d[vid] for vid in F]) for F in faces
    ])
    print("\tUnion: ", time.perf_counter() - s, "s")

    return [P for P in shapely.get_parts(Union) if isinstance(P, shapely.Polygon)]

############################################################################################################################
# Point Cloud
############################################################################################################################
def isSymmetricAlong(pcd: o3d.geometry.PointCloud, axis: Literal['x', 'y', 'z'], thresh: float):
    """ 檢查 pcd 沿著某一個軸是否是對稱的 """
    axis = {'x': 0, 'y': 1, 'z': 2}[axis]
    points = np.asarray(pcd.points).copy()
    points[:] -= pcd.get_center() # 平移使得中心在 (0, 0, 0)

    # 將點雲切兩半，分成 <= 0 和 > 0
    points1 = points[points[:, axis] <= 0]
    points2 = points[points[:, axis] > 0]

    # 對 points1 沿著 axis 軸鏡像
    points1[:, axis] = -points1[:, axis]

    # 如果距離夠小代表對稱
    distanceVector = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1)).compute_point_cloud_distance(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2))
    )
    mean_dist = np.asarray(distanceVector).mean()

    return mean_dist <= thresh

def adjustCenterInPlace(pcd: o3d.geometry.PointCloud, center: list[float]):
    """ 如果 pcd 沿著 X or Z 軸對稱，則將 center 移動到 X or Z 的中點 """
    if isSymmetricAlong(pcd, 'x', 0.01):
        center[0] = pcd.get_center()[0]
    if isSymmetricAlong(pcd, 'z', 0.01):
        center[2] = pcd.get_center()[2]

def AddBoundaryWeight(pcd: o3d.geometry.PointCloud, silhouette: shapely.Polygon, *, dist: float = 0.01):
    """
    將靠近 silhouette 邊界上的點加重權重
    
    :param dist: 靠近邊界 dist 以內的點會複製一份
    """
    lines: list[shapely.LinearRing] = [silhouette.exterior]
    lines += [inRing for inRing in silhouette.interiors]
    points_3d = [shapely.Point(p[0], p[2], p[1]) for p in pcd.points]

    tree = STRtree(lines)
    point_indices = list(set(tree.query(points_3d, 'dwithin', dist)[0].tolist()))

    points_3d = np.asarray(points_3d)
    points_3d = np.hstack([
        points_3d,
        points_3d[point_indices]
    ])

    pcd.points = o3d.utility.Vector3dVector([(p.coords[0][0], p.coords[0][2], p.coords[0][1]) for p in points_3d])

############################################################################################################################
# Reference: https://stackoverflow.com/a/75086582/20876404
############################################################################################################################
def sliceplane(mesh: o3d.geometry.TriangleMesh, axis, value, direction):
    # axis can be 0,1,2 (which corresponds to x,y,z)
    # value where the plane is on that axis
    # direction can be True or False (True means remove everything that is
    # greater, False means less
    # than)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    new_vertices = list(vertices)
    new_triangles = []

    # (a, b) -> c
    # c refers to index of new vertex that sits at the intersection between a,b
    # and the boundingbox edge
    # a is always inside and b is always outside
    intersection_edges = dict()

    # find axes to compute
    axes_compute = [0,1,2]
    # remove axis that the plane is on
    axes_compute.remove(axis)

    def compute_intersection(vertex_in_index, vertex_out_index):
        vertex_in = vertices[vertex_in_index]
        vertex_out = vertices[vertex_out_index]
        if (vertex_in_index, vertex_out_index) in intersection_edges:
            intersection_index = intersection_edges[(vertex_in_index, vertex_out_index)]
            intersection = new_vertices[intersection_index]
        else:
            intersection = [None, None, None]
            intersection[axis] = value
            const_1 = (value - vertex_in[axis])/(vertex_out[axis] - vertex_in[axis])
            c = axes_compute[0]
            intersection[c] = (const_1 * (vertex_out[c] - vertex_in[c])) + vertex_in[c]
            c = axes_compute[1]
            intersection[c] = (const_1 * (vertex_out[c] - vertex_in[c])) + vertex_in[c]
            assert not (None in intersection)
            # save new vertice and remember that this intersection already added an edge
            new_vertices.append(intersection)
            intersection_index = len(new_vertices) - 1
            intersection_edges[(vertex_in_index, vertex_out_index)] = intersection_index

        return intersection_index

    for t in triangles:
        v1, v2, v3 = t
        if direction:
            v1_out = vertices[v1][axis] > value
            v2_out = vertices[v2][axis] > value
            v3_out = vertices[v3][axis] > value
        else: 
            v1_out = vertices[v1][axis] < value
            v2_out = vertices[v2][axis] < value
            v3_out = vertices[v3][axis] < value

        bool_sum = sum([v1_out, v2_out, v3_out])
        # print(f"{v1_out=}, {v2_out=}, {v3_out=}, {bool_sum=}")

        if bool_sum == 0:
            # triangle completely inside --> add and continue
            new_triangles.append(t)
        elif bool_sum == 3:
            # triangle completely outside --> skip
            continue
        elif bool_sum == 2:
            # two vertices outside 
            # add triangle using both intersections
            vertex_in_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
            vertex_out_1_index = v1 if v1_out else (v2 if v2_out else v3)
            vertex_out_2_index = v3 if v3_out else (v2 if v2_out else v1)
            # print(f"{vertex_in_index=}, {vertex_out_1_index=}, {vertex_out_2_index=}")
            # small sanity check if indices sum matches
            assert sum([vertex_in_index, vertex_out_1_index, vertex_out_2_index]) == sum([v1,v2,v3])

            # add new triangle 
            new_triangles.append([vertex_in_index, compute_intersection(vertex_in_index, vertex_out_1_index), 
                compute_intersection(vertex_in_index, vertex_out_2_index)])

        elif bool_sum == 1:
            # one vertice outside
            # add three triangles
            vertex_out_index = v1 if v1_out else (v2 if v2_out else v3)
            vertex_in_1_index = v1 if (not v1_out) else (v2 if (not v2_out) else v3)
            vertex_in_2_index = v3 if (not v3_out) else (v2 if (not v2_out) else v1)
            # print(f"{vertex_out_index=}, {vertex_in_1_index=}, {vertex_in_2_index=}")
            # small sanity check if outdices sum matches
            assert sum([vertex_out_index, vertex_in_1_index, vertex_in_2_index]) == sum([v1,v2,v3])

            new_triangles.append([vertex_in_1_index, compute_intersection(vertex_in_1_index, vertex_out_index), vertex_in_2_index])
            new_triangles.append([compute_intersection(vertex_in_1_index, vertex_out_index), 
                compute_intersection(vertex_in_2_index, vertex_out_index), vertex_in_2_index])

        else:
            assert False

    # TODO remap indices and remove unused 

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(new_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
    return mesh

def clean_crop_aabb(mesh: o3d.geometry.TriangleMesh, min_corner, max_corner):
    """ 對 Triangle Mesh 切割，切出 AABB 範圍內的 mesh """
    min_x = min(min_corner[0], max_corner[0])
    min_y = min(min_corner[1], max_corner[1])
    min_z = min(min_corner[2], max_corner[2])
    max_x = max(min_corner[0], max_corner[0])
    max_y = max(min_corner[1], max_corner[1])
    max_z = max(min_corner[2], max_corner[2])

    # mesh = sliceplane(mesh, 0, min_x, False)
    if np.isfinite(max_x): mesh_sliced = sliceplane(mesh, 0, max_x, True)
    if np.isfinite(min_x): mesh_sliced = sliceplane(mesh_sliced, 0, min_x, False)
    if np.isfinite(max_y): mesh_sliced = sliceplane(mesh_sliced, 1, max_y, True)
    if np.isfinite(min_y): mesh_sliced = sliceplane(mesh_sliced, 1, min_y, False)
    if np.isfinite(max_z): mesh_sliced = sliceplane(mesh_sliced, 2, max_z, True)
    if np.isfinite(min_z): mesh_sliced = sliceplane(mesh_sliced, 2, min_z, False)
    # mesh_sliced = mesh_sliced.paint_uniform_color([0,0,1])

    return mesh_sliced
