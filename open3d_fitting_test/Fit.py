import pyransac3d as pyrsc
import numpy as np
import shapely
import open3d as o3d

from .Util import shapely_poly_to_open3d_mesh, adjustCenterInPlace, clean_crop_aabb

def fitPlaneRANSAC(points: np.ndarray, silhouette: shapely.Polygon | shapely.MultiPolygon) -> o3d.geometry.TriangleMesh:
    """
    將點雲 fitting 成平面然後將 silhouette 的區塊沿 Y 軸方向投影上去
    """
    eq_P, _ = pyrsc.Plane().fit(points)

    mesh = shapely_poly_to_open3d_mesh(silhouette)

    # project alphashape
    vert = np.asarray(mesh.vertices)
    for i in range(len(vert)):
        vert[i, 1] = -(eq_P[0] * vert[i, 0] + eq_P[2] * vert[i, 2] + eq_P[3]) / eq_P[1]

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
    for i in range(len(vert)):
        vert[i] = vert[i] + center

    # 切除
    max_bound = aabb.get_max_bound()
    max_bound[1] = np.inf # 高度不切最高
    mesh = clean_crop_aabb(mesh, aabb.get_min_bound(), max_bound)

    return mesh
