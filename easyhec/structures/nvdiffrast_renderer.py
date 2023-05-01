import matplotlib.pyplot as plt
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh

from easyhec.utils.nvdiffrast_utils import K_to_projection, transform_pos


class NVDiffrastRenderer:
    def __init__(self, image_size):
        """
        image_size: H,W
        """
        # self.
        self.H, self.W = image_size
        self.resolution = image_size
        blender2opencv = torch.tensor([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]]).float().cuda()
        self.opencv2blender = torch.inverse(blender2opencv)
        self.glctx = dr.RasterizeCudaContext()

    def render_mask(self, verts, faces, K, object_pose, anti_aliasing=True):
        """
        @param verts: N,3, torch.tensor, float, cuda
        @param faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        @param object_pose: 4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        proj = K_to_projection(K, self.H, self.W)

        pose = self.opencv2blender @ object_pose

        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=self.resolution)
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask

    def batch_render_mask(self, verts, faces, K, anti_aliasing=True):
        """
        @param batch_verts: N,3, torch.tensor, float, cuda
        @param batch_faces: M,3, torch.tensor, int32, cuda
        @param K: 3,3 torch.tensor, float ,cuda
        # @param batch_object_poses: N,4,4 torch.tensor, float, cuda
        @return: mask: 0 to 1, HxW torch.cuda.FloatTensor
        """
        proj = K_to_projection(K, self.H, self.W)

        pose = self.opencv2blender

        pos_clip = transform_pos(proj @ pose, verts)

        rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=self.resolution)
        if anti_aliasing:
            vtx_color = torch.ones(verts.shape, dtype=torch.float, device=verts.device)
            color, _ = dr.interpolate(vtx_color[None, ...], rast_out, faces)
            color = dr.antialias(color, rast_out, pos_clip, faces)
            mask = color[0, :, :, 0]
        else:
            mask = rast_out[0, :, :, 2] > 0
        mask = torch.flip(mask, dims=[0])
        return mask


def main():
    pose = np.array([[0.99638397, -0.0846324, 0.00750877, -0.20668708],
                     [-0.00875172, -0.19013488, -0.9817189, 0.08405855],
                     [0.0845129, 0.97810328, -0.19018805, 0.77892876],
                     [0., 0., 0., 1.]]).astype(np.float32)
    pose = torch.from_numpy(pose).cuda()
    pose.requires_grad = True
    mesh = trimesh.load_mesh("data/xarm_description/meshes/xarm7/visual/link0.STL")
    K = np.loadtxt("data/realsense/20230124_092547/K.txt")
    H, W = 720, 1280
    renderer = NVDiffrastRenderer([H, W])
    mask = renderer.render_mask(torch.from_numpy(mesh.vertices).cuda().float(),
                                torch.from_numpy(mesh.faces).cuda().int(),
                                torch.from_numpy(K).cuda().float(),
                                pose)
    plt.imshow(mask.detach().cpu())
    plt.show()


if __name__ == '__main__':
    main()
