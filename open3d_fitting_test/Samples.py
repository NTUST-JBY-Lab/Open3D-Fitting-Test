"""
取樣變點雲
"""
import open3d as o3d
import numpy as np

def sample_mesh_with_raycast(mesh: o3d.geometry.TriangleMesh, step=0.01) -> tuple[np.ndarray, np.ndarray]:
    """
    :returns (hit_points, hit_normals): 
    """
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
    
def sample_along_edges(mesh: o3d.geometry.TriangleMesh, interval: float) -> np.ndarray:
    """
    對 mesh 的每個邊做取樣
    """
    # Get all unique edge pairs
    F = np.asarray(mesh.triangles) # N * 3
    E = np.sort(np.vstack([
        np.column_stack((F[:, 0], F[:, 1])),
        np.column_stack((F[:, 1], F[:, 2])),
        np.column_stack((F[:, 2], F[:, 0]))
    ]), axis=1)
    E = np.unique(E, axis=0)

    points = []
    # Sample Along Edges
    for v1, v2 in E:
        points += sample_edge(mesh, v1, v2, interval)

    if len(points) > 0:
        return np.vstack([np.array(points), mesh.vertices])
    else:
        return np.asarray(mesh.vertices)
