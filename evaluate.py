import numpy as np
import trimesh
from scipy.spatial import cKDTree

POINTS_TO_SAMPLE = None

def load_and_sample_mesh(filename):
    # Load mesh from a .obj file
    mesh = trimesh.load(filename, process=True)
    points, _ = trimesh.sample.sample_surface(mesh, POINTS_TO_SAMPLE)
    return points

def load_and_prepare_dat(filename):
    # Load .dat file (pickled numpy array, bytes encoding)
    points = np.load(filename, allow_pickle=True, encoding='bytes')
    POINTS_TO_SAMPLE = points.shape[0]
    return points[:, :3]

def load_dat_and_obj(dat_filename, obj_filename):
    ground_truth_points = load_and_prepare_dat(dat_filename)
    generated_points = load_and_sample_mesh(obj_filename)
    return ground_truth_points, generated_points

def normalize_points(points):
    # Normalize points into the cube [-1, 1]^3
    min_val = np.min(points, axis=0)
    max_val = np.max(points, axis=0)
    points = 2 * (points - min_val) / (max_val - min_val) - 1
    return points

def chamfer_distance(p1, p2):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    closest_p2_to_p1, _ = tree1.query(p2)
    closest_p1_to_p2, _ = tree2.query(p1)
    cd = np.mean(closest_p2_to_p1) + np.mean(closest_p1_to_p2)
    return cd / 2

def f_score(p1, p2, threshold=1e-4):
    tree1 = cKDTree(p1)
    tree2 = cKDTree(p2)
    recall = np.mean(tree1.query(p2)[0] < threshold)
    precision = np.mean(tree2.query(p1)[0] < threshold)
    if recall + precision == 0:
        return 0
    fs = 2 * precision * recall / (precision + recall)
    return fs

# Paths to your files
# using example plane obj for proof of concept
obj_filename = '/om/user/evan_kim/InstantMesh/outputs/instant-mesh-large/meshes/02691156.obj'
dat_filename = '/om/user/evan_kim/SculptFormer/datasets/data/shapenet/data_tf/02691156/98b163efbbdf20c898dc7d57268f30d4/rendering/03.dat'

# Load and sample meshes
ground_truth_points, generated_points = load_dat_and_obj(dat_filename, obj_filename)

# Normalize points
generated_points = normalize_points(generated_points)
ground_truth_points = normalize_points(ground_truth_points)

# Compute metrics
cd = chamfer_distance(generated_points, ground_truth_points)
fs = f_score(generated_points, ground_truth_points)

print("Chamfer Distance:", cd)
print("F-Score:", fs)
