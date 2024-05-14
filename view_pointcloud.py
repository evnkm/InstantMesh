import numpy as np
import trimesh

PHOTO_IDX = 3

DAT_FILE = f"/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/0{PHOTO_IDX}.dat"
OBJ_FILE = "/om/user/evan_kim/InstantMesh/outputs/instant-mesh-large/meshes/02691156.obj"
RENDERING_METADATA = "/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/rendering_metadata.txt"
SCALING_FACTOR = 0.19

# new_order = [0, 2, 1]
# new_order = [1, 0, 2]
# new_order = [1, 2, 0]
# new_order = [2, 0, 1]
# new_order = [2, 1, 0]

def inverse_transform(train_data, param):
    # Unpack camera parameters
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])
    camY = param[3] * np.sin(phi)
    temp = param[3] * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    # Compute camera matrix
    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)
    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])

    # Extract transformed positions and normals
    pt_trans = train_data[:, :3]
    nom_trans = train_data[:, 3:]

    # Inverse transformation for positions
    position = np.dot(pt_trans, cam_mat) + cam_pos

    # Inverse transformation for normals
    # normal = np.dot(nom_trans, cam_mat)

    return position #, normal

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


render_meta = np.loadtxt(RENDERING_METADATA)
dat_points = np.load(DAT_FILE, allow_pickle=True, encoding='bytes')
param = render_meta[PHOTO_IDX]

dat_points = inverse_transform(dat_points, param) / SCALING_FACTOR
# shift the points to the origin
dat_points -= np.mean(dat_points, axis=0)
point_cloud_dat = trimesh.points.PointCloud(dat_points, colors=[255, 0, 0])

mesh = trimesh.load(OBJ_FILE, process=True)
mesh_points, _ = trimesh.sample.sample_surface(mesh, 8179)
# shift the points to the origin
mesh_points -= np.mean(mesh_points, axis=0)
point_cloud_mesh = trimesh.points.PointCloud(mesh_points, colors=[0, 255, 0])

# Visualize the point clouds
scene = trimesh.Scene()
scene.add_geometry(point_cloud_dat)
scene.add_geometry(point_cloud_mesh)
scene.show()