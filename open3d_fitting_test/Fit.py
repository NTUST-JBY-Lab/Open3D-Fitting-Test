import pyransac3d as pyrsc
import numpy as np
import shapely
import open3d as o3d

from .Util import shapely_poly_to_open3d_mesh, adjustCenterInPlace, clean_crop_aabb, AddBoundaryWeight

def fitPlaneRANSAC(points: np.ndarray, silhouette: shapely.Polygon | shapely.MultiPolygon) -> o3d.geometry.TriangleMesh:
    """
    將點雲 fitting 成平面然後將 silhouette 的區塊沿 Y 軸方向投影上去
    """
    # 轉點雲
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    # 邊界加厚
    AddBoundaryWeight(pcd, silhouette)
    # 估計 Normal
    pcd.estimate_normals()
    # RANSAC
    eq_P, _ = pcd.segment_plane(0.01, 3, 1000)

    mesh = shapely_poly_to_open3d_mesh(silhouette)

    # project silhouette
    vert = np.asarray(mesh.vertices)
    vert[:, 1] = -(eq_P[0] * vert[:, 0] + eq_P[2] * vert[:, 2] + eq_P[3]) / eq_P[1]

    return mesh

def fitSphereRANSAC(points: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    將點雲 fitting 成球
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    aabb = pcd.get_axis_aligned_bounding_box()

    center, radius, _ = pyrsc.Sphere().fit(points)
    adjustCenterInPlace(pcd, center)

    # 建立以 center 為球心，半徑 radius 的球
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=10)
    vert = np.asarray(mesh.vertices)
    vert[:] = vert[:] + center

    # 切除
    max_bound = aabb.get_max_bound()
    max_bound[1] = np.inf # 高度不切最高
    mesh = clean_crop_aabb(mesh, aabb.get_min_bound(), max_bound)

    return mesh
