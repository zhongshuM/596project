import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import os


num = 5
indir = './results'
outdir = './ply'

if not os.path.exists(outdir):
	os.makedirs(outdir)


print(f'image {48}')
color_raw = o3d.io.read_image(f'{indir}/{48}_orign.png')
depth_raw = o3d.geometry.Image((100000.0 / np.asarray(o3d.io.read_image(f'{indir}/{48}_depth.png'))).astype(np.uint16))
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

inter = o3d.camera.PinholeCameraIntrinsic()
# inter.set_intrinsics(640, 480, 518.857901, 519.469611, 284.582449, 208.736166)
inter.set_intrinsics(640, 480, 518.857901, 519.469611, 325.582449, 253.736166)
pcd = o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image, inter)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(f'{outdir}/{i}.ply', pcd)
