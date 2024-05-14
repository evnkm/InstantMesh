import numpy as np
import trimesh
dat_points = np.load("/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/03.dat", allow_pickle=True, encoding='bytes')
dat_points = dat_points[:, :3] / 0.19
# shift the points to the origin
dat_points -= np.mean(dat_points, axis=0)

# new_order = [0, 2, 1]
# new_order = [1, 0, 2]
# new_order = [1, 2, 0]
# new_order = [2, 0, 1]
# new_order = [2, 1, 0]

point_cloud_dat = trimesh.points.PointCloud(dat_points, colors=[255, 0, 0])

mesh = trimesh.load("/om/user/evan_kim/InstantMesh/outputs/instant-mesh-large/meshes/02691156.obj", process=True)
mesh_points, _ = trimesh.sample.sample_surface(mesh, 8179)
# shift the points to the origin
mesh_points -= np.mean(mesh_points, axis=0)

point_cloud_mesh = trimesh.points.PointCloud(mesh_points, colors=[0, 255, 0])

# Visualize the point clouds
scene = trimesh.Scene()
scene.add_geometry(point_cloud_dat)
scene.add_geometry(point_cloud_mesh)
scene.show()