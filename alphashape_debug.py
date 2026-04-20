"""
alphashape.alphashape 在找輪廓時可能會變成一條線，應該是 Polygonize 那段有問題
"""
import open3d as o3d
from scipy.spatial import Delaunay
import numpy as np
import alphashape
import shapely
import pandas as pd
from Util import shapely_poly_to_open3d_mesh

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