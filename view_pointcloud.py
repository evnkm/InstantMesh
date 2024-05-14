import numpy as np
import trimesh
dat_points = np.load("/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/03.dat", allow_pickle=True, encoding='bytes')
dat_points = dat_points[:, :3] / 0.57
point_cloud_dat = trimesh.points.PointCloud(dat_points)

mesh = trimesh.load("/om/user/evan_kim/InstantMesh/outputs/instant-mesh-large/meshes/02691156.obj", process=True)
mesh_points, _ = trimesh.sample.sample_surface(mesh, 8179)
point_cloud_mesh = trimesh.points.PointCloud(mesh_points)


pc1 = trimesh.points.PointCloud(point_cloud_dat, colors=[1.0, 0.0, 0.0])
pc2 = trimesh.points.PointCloud(point_cloud_mesh, colors=[0.0, 1.0, 0.0])

# Visualize the point clouds
scene = trimesh.Scene()
scene.add_geometry(pc1)
scene.add_geometry(pc2)
scene.show()