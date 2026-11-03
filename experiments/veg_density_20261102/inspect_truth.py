"""What the Brandenburg LiDAR arrays hold, and how well they line up."""
import numpy as np, glob, os
root = r"E:\Ortho4XP_Review\canopy_height\truth"
for name in ("lidar_schulz", "lidartop_schulz", "lidar_zeuthen", "lidar_app19", "lidartop_app19", "ndsm_z18"):
    a = np.load(os.path.join(root, name + ".npy"))
    print(name, a.dtype, a.shape, "nan", float(np.isnan(a).mean()), "p50/90/99", np.nanpercentile(a, [50, 90, 99]), "share>=2", float((a >= 2).mean()))
for name in ("canopymaps_schulz", "canopymaps_zeuthen"):
    a = np.load(os.path.join(r"E:\Ortho4XP_Review\canopy_height", name + ".npy"))
    print(name, a.dtype, a.shape)
print(open(os.path.join(root, "bboxes.txt")).read())
