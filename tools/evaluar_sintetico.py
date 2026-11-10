#!/usr/bin/env python3
"""Compara una nube procesada (nube_fusionada.ply, Z arriba) contra la escena real del simulador.
Uso: python evaluar_sintetico.py <carpeta_resultado> [escena_real.ply]"""
import sys, numpy as np, open3d as o3d
from scipy.optimize import least_squares
d=sys.argv[1]
gt=o3d.io.read_triangle_mesh(sys.argv[2] if len(sys.argv)>2 else 'escena_real.ply')
sc=o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
ZtoY=np.linalg.inv(np.array([[1,0,0],[0,0,-1],[0,1,0]],float))
P=np.asarray(o3d.io.read_point_cloud(d+'/nube_fusionada.ply').points)@ZtoY.T
e=sc.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy()
f=P[(np.abs(P[:,1])<0.03)&((np.abs(P[:,0])>1.35)|(P[:,2]>0.7))&(P[:,2]>-1.5)]
tgt=gt.sample_points_uniformly(600000); tgt.estimate_normals()
src=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.01)
r=o3d.pipelines.registration.registration_icp(src,tgt,0.02,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPlane())
def pipe(yc,zc,R,od):
    rr=np.hypot(P[:,1]-yc,P[:,2]-zc); m=(np.abs(P[:,0])<1)&(np.abs(rr-R)<0.02); q=P[m][:,1:]
    x=least_squares(lambda p: np.hypot(q[:,0]-p[0],q[:,1]-p[1])-p[2],[yc,zc,R]).x
    return x[2]*2000-od, np.hypot(x[0]-yc,x[1]-zc)*1000
p2=pipe(1.05,0.25,0.0254,50.8); p3=pipe(1.20,-0.10,0.0381,76.2)
print(f"{d:22s} pts {len(P)/1e6:4.2f}M  mediana {np.median(e)*1000:4.2f}  95% {np.percentile(e,95)*1000:5.2f} mm | piso {f[:,1].mean()*1000:+5.2f} | desfase ICP {np.linalg.norm(r.transformation[:3,3])*1000:4.2f} | 2\" Ø{p2[0]:+4.1f} eje {p2[1]:3.1f} | 3\" Ø{p3[0]:+4.1f} eje {p3[1]:3.1f}")
