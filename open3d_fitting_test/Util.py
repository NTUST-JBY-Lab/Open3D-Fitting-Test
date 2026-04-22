import shapely
import numpy as np
import open3d as o3d
import alphashape

def shapely_poly_to_open3d_mesh(poly: shapely.Polygon, y_value=0.0):
    """
    將 Shapely Polygon 轉換為 Open3D TriangleMesh
    :param poly: shapely.geometry.Polygon 物件
    :param y_value: 投影到 3D 空間時的 Y 座標 (或是 Z 座標，依你的坐標系而定)
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
            pt = (coord[0], y_value, coord[1]) # 假設你之前是投影到 XZ 平面
            
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

def alaphashape_union(points_2d: list[tuple[float, float]], *, alpha: float = 50) -> list[shapely.Polygon]:
    """
    三角化 -> 留外切圓半徑夠小的 -> Union
    """
    if alpha == 0:
        return [shapely.MultiPoint(points_2d).convex_hull]

    faces = []
    # 對點雲做 Delaunay 然後計算每個面的外接圓半徑
    for simplex, radius in alphashape.alphasimplices(points_2d):
        # radius 夠小 -> 留下
        if radius < 1 / alpha:
            face = shapely.Polygon([points_2d[vid] for vid in simplex])
            if face.is_valid:
                faces.append(face)

    # 將所有留下的面做 Union
    Union = shapely.unary_union(faces)

    return [P for P in shapely.get_parts(Union) if isinstance(P, shapely.Polygon)]

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
    min_x = min(min_corner[0], max_corner[0])
    min_y = min(min_corner[1], max_corner[1])
    min_z = min(min_corner[2], max_corner[2])
    max_x = max(min_corner[0], max_corner[0])
    max_y = max(min_corner[1], max_corner[1])
    max_z = max(min_corner[2], max_corner[2])

    # mesh = sliceplane(mesh, 0, min_x, False)
    mesh_sliced = sliceplane(mesh, 0, max_x, True)
    mesh_sliced = sliceplane(mesh_sliced, 0, min_x, False)
    mesh_sliced = sliceplane(mesh_sliced, 1, max_y, True)
    mesh_sliced = sliceplane(mesh_sliced, 1, min_y, False)
    mesh_sliced = sliceplane(mesh_sliced, 2, max_z, True)
    mesh_sliced = sliceplane(mesh_sliced, 2, min_z, False)
    # mesh_sliced = mesh_sliced.paint_uniform_color([0,0,1])

    return mesh_sliced
