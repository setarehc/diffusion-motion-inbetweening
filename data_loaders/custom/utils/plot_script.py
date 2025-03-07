import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def test_plot_circle():
    # matplotlib.use('Agg')
    fig = plt.figure(figsize=(3, 3))
    plt.tight_layout()
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")

    x_c = 1
    y_c = 0.1
    z_c = 1
    r = 2
    
    theta = np.linspace(0, 2 * np.pi, 300) # 300 points on the circle
    x = x_c + r * np.sin(theta)
    y = y_c + theta * 0.0
    z = z_c + r * np.cos(theta)
    import pdb; pdb.set_trace()
    ax.plot3D(x, y, z, color="red")
    plt.show()
    
    return


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], traj_only=False, target_pose=None, kframes=[], obs_list=[]):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
    
    def plot_trajectory(trajec_idx):
        ax.plot3D([0 - trajec_idx[0], 0 - trajec_idx[0]], [0.2, 0.2], [0 - trajec_idx[1], 1 - trajec_idx[1]], color="red") # (x,y,z)
    
    def plot_ref_axes(trajec_idx):
        '''
        trajec_idx contains (x,z) coordinate of the root of the current frame.
        Need to offset the reference axes because the plot is root-centered
        '''
        ax.plot3D([0 - trajec_idx[0], 0 - trajec_idx[0]], [0.2, 0.2], [0 - trajec_idx[1], 1 - trajec_idx[1]], color="red") # (x,y,z)
        ax.plot3D([0 - trajec_idx[0], 1 - trajec_idx[0]], [0.2, 0.2], [0 - trajec_idx[1], 0 - trajec_idx[1]], color="yellow") # (x,y,z)

    def plot_ground_target(trajec_idx):
        # kframes = [(30,  (0.0, 3.0)),
        #             (45,  (1.5, 3.0)),
        #             (60,  (3.0, 3.0)),
        #             (75,  (3.0, 1.5)),
        #             (90,  (3.0, 0.0)),
        #             (105, (1.5, 0.0)),
        #             (120, (0.0, 0.0))
        #             ]
        pp = [(bb[0] * 1.3, bb[1] * 1.3) for (aa, bb) in kframes]
        for i in range(len(pp)):
            ax.plot3D([pp[i][0] - trajec_idx[0], pp[i][0] - trajec_idx[0]], [0.0, 0.1], [pp[i][1] - trajec_idx[1], pp[i][1] - trajec_idx[1]], color="blue") # (x,y,z)
    
    def plot_obstacles(trajec_idx):
        for i in range(len(obs_scale)):
            x_c = obs_scale[i][0][0] - trajec_idx[0]
            y_c = 0.1
            z_c = obs_scale[i][0][1] - trajec_idx[1]
            r = obs_scale[i][1]
            # Draw circle
            theta = np.linspace(0, 2 * np.pi, 300) # 300 points on the circle
            x = x_c + r * np.sin(theta)
            y = y_c + theta * 0.0
            z = z_c + r * np.cos(theta)
            ax.plot3D(x, y, z, color="red") # linewidth=2.0

    def plot_target_pose(target_pose, frame_idx, cur_root_loc, used_colors, kinematic_tree):
        # The target pose is re-centered in every frame because the plot is root-centered
        # used_colors = colors_blue if index in gt_frames else colors
        for target_frame in frame_idx:
            for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                # print("i = ", i, data[index, chain, 0], data[index, chain, 1], data[index, chain, 2])
                ax.plot3D(target_pose[target_frame, chain, 0] - cur_root_loc[0],
                          target_pose[target_frame, chain, 1],
                          target_pose[target_frame, chain, 2] - cur_root_loc[2],
                          linewidth=linewidth, color=color)
    

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    if target_pose is None:
        target_pose = np.zeros_like(data)

    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
        target_pose *= 0.003
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
        target_pose *= 1.3
        obs_scale = [((loc[0] * 1.3, loc[1] * 1.3), rr * 1.3) for (loc, rr) in obs_list]
    elif dataset in ['humanact12', 'uestc']:
        data *= -1.5 # reverse axes, scale for visualization
        target_pose *= -1.5

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    # ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection="3d")
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'lower_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    target_pose[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    # Data is root-centered in every frame
    data_copy = data.copy()
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    # Center first frame of target pose
    # target_pose[:, :, 0] -= data_copy[0:1, :, 0]
    # target_pose[:, :, 2] -= data_copy[0:1, :, 2]

    #     print(trajec.shape)

    def update(index):
        ax.clear()
        #         print(index)
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        plot_obstacles(trajec[index])
        plot_ground_target(trajec[index])

        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        # TODO: if GMD:
        # Now only use orange color. Blue color is used for ground truth condition
        # used_colors = colors_orange
        
        used_colors = colors_blue if index in gt_frames else colors
        
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            # print("i = ", i, data[index, chain, 0], data[index, chain, 1], data[index, chain, 2])
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)
        if traj_only:
            ax.scatter(data[index, 0, 0], data[index, 0, 1], data[index, 0, 2], color=color)
        # Test plot trajectory
        # plot_trajectory(trajec[index])

        def plot_root_horizontal():
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 1]), trajec[:index, 1] - trajec[index, 1], linewidth=2.0,
                      color=used_colors[0])

        # plot_ref_axes(trajec[index])
        
        plot_root_horizontal()
        
        
        plot_target_pose(target_pose, gt_frames, data_copy[index, 0, :], colors_blue, kinematic_tree)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()
