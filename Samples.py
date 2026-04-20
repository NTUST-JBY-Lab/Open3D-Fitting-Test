"""
取樣變點雲
"""
import open3d as o3d
import numpy as np

def sample_mesh_with_raycast(mesh: o3d.geometry.TriangleMesh, step=0.01):
    # 1. 建立 Tensor 場景
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    # 2. 取得邊界範圍
    bbox = mesh.get_axis_aligned_bounding_box()
    min_b, max_b = bbox.get_min_bound(), bbox.get_max_bound()

    # 3. 生成 XZ 射線網格
    x_range = np.arange(min_b[0], max_b[0], step)
    z_range = np.arange(min_b[2], max_b[2], step)
    grid_x, grid_z = np.meshgrid(x_range, z_range)
    x_flat, z_flat = grid_x.flatten(), grid_z.flatten()
    
    # 4. 準備射線 Tensor [N, 6]
    start_y = max_b[1] + 1.0
    rays = np.zeros((len(x_flat), 6), dtype=np.float32)
    rays[:, 0], rays[:, 1], rays[:, 2] = x_flat, start_y, z_flat
    rays[:, 4] = -1  # 方向朝下
    
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    # 5. 執行 Raycasting
    ans = scene.cast_rays(rays_t)
    
    # 6. 提取與過濾
    t_hit = ans['t_hit'].numpy()
    normals_all = ans['primitive_normals'].numpy() # 取得所有射線對應的面法向量
    mask = np.isfinite(t_hit)

    # 計算擊中點位置
    hit_points = np.zeros((np.sum(mask), 3))
    hit_points[:, 0] = x_flat[mask]
    hit_points[:, 1] = start_y - t_hit[mask]
    hit_points[:, 2] = z_flat[mask]

    # 取得對應點的法向量
    hit_normals = normals_all[mask]

    return hit_points, hit_normals


def ordered_tuple(a: float, b: float):
    """
    讓 tuple 變有序的
    """
    return (a, b) if a <= b else (b, a)

def lerp(p1: tuple[float, float, float], p2: tuple[float, float, float], alpha: float):
    """
    線性內插兩個座標點
    """
    return tuple(map(
        lambda pair: pair[0] + (pair[1] - pair[0]) * alpha,  # 對每個分量內插
        zip(p1, p2) # 變 [(x1, x2), (y1, y2), (z1, z2)]
    ))

def sample_edge(mesh: o3d.geometry.TriangleMesh, v1: int, v2: int, interval: float) -> list[tuple[float, float, float]]:
    """
    對 mesh 的 (v1, v2) 這個邊做取樣，取樣間隔為 interval (不含v1, v2)
    """
    start, end = tuple(mesh.vertices[v1]), tuple(mesh.vertices[v2])
    dist = np.linalg.norm(mesh.vertices[v1] - mesh.vertices[v2])
    accum = interval

    result = []

    # 不斷向前 interval 的距離然後取樣
    while accum < dist:
        result.append(lerp(start, end, accum / dist))
        accum += interval

    return result
    
def sample_along_edges(mesh: o3d.geometry.TriangleMesh, interval: float) -> o3d.geometry.PointCloud:
    """
    對 mesh 的每個邊做取樣
    """
    points = []
    visited_edge = set()
    visited_vert = set()

    # Sample Along Edges
    for face in mesh.triangles:
        # 如果頂點未拜訪過，加入座標
        for i in range(3):
            v = face[i]

            if not v in visited_vert:
                points.append(tuple(mesh.vertices[v]))
                visited_vert.add(v)

        # 對三個邊取樣
        for i in range(3):
            vStart, vEnd = ordered_tuple(face[i], face[(i + 1) % 3])

            if not (vStart, vEnd) in visited_edge:
                points += sample_edge(mesh, vStart, vEnd, interval)
                visited_edge.add((vStart, vEnd))

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(set(points)))
