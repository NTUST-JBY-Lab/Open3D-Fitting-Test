"""
Note:
- 本函式庫所有函式假設 TriangleMesh 是 y 軸向上，使用右手座標系。
- 而 shapely.Polygon 則是 z 軸向上（照 shapely 的慣例），左手座標系。
- 
- 在 shapely_poly_to_open3d_mesh 中， shpely.Polygon -> TriangleMesh 的方式為 Y Z 軸互換
"""
from .Fit import fitCreatePlaneRANSAC, fitCreateSphereRANSAC
from .Samples import sample_mesh_with_raycast, sample_along_edges
from .Util import alaphashape_union2D, clean_crop_aabb, isSymmetricAlong, shapely_poly_to_open3d_mesh
