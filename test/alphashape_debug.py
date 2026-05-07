"""
alphashape.alphashape 在找輪廓時可能會變成一條線，應該是 Polygonize 那段有問題
"""
import open3d as o3d
from scipy.spatial import Delaunay
import numpy as np
import alphashape
import shapely
import os
from open3d_fitting_test.Util import shapely_poly_to_open3d_mesh, alaphashape_union2D

PLY_FILE = os.path.join(os.path.dirname(__file__), "alphashape_debug/0_pcd.ply")

def alphashape1():
    """
    三角化 -> 留外切圓半徑夠小的
    """
    pcd = o3d.io.read_point_cloud(PLY_FILE)

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
    """
    三角化 -> 留外切圓半徑夠小的 -> Union -> 留 exterior -> simplify
    """
    from shapely.plotting import plot_polygon
    import matplotlib.pyplot as plt
    
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    points_2d = [(x, z) for x, y, z in np.asarray(pcd.points)]

    polygons = [P.simplify(0.01).buffer(0) for P in alaphashape_union2D(points_2d) if P.is_valid]

    for i, P in enumerate(polygons):
        plot_polygon(P)
        print("is_simple:", P.is_simple, "is_valid:", P.is_valid)
        plt.title(f"{i + 1} / {len(polygons)}")
        plt.show()

    mesh = shapely_poly_to_open3d_mesh(shapely.MultiPolygon(polygons))
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, mesh_show_back_face=True)
    o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(__file__), "alphashape_debug/alphashape.obj"), mesh)
    
def alphashape3():
    """
    使用 alphashape.alphashape
    """
    pcd = o3d.io.read_point_cloud(PLY_FILE)
    points_2d = [(x, z) for x, y, z in np.asarray(pcd.points)]
    
    shapes = alphashape.alphashape(points_2d, 50)

    o3d.visualization.draw_geometries([shapely_poly_to_open3d_mesh(shapes)], mesh_show_wireframe=True, mesh_show_back_face=True)

if __name__ == "__main__":
    alphashape1()
    alphashape2()
    alphashape3()