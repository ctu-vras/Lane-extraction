#from mayavi import mlab
#import mayavi.mlab as mlab
from matplotlib import animation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
def visualize_points(points_before:np.ndarray,points:np.ndarray, vis_num_tresh = 500000):
    """fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def init():
        ax.scatter(points_before[:, 0], points_before[:, 1], points_before[:, 2], color='grey', s=0.02, alpha=0.1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=0.02, alpha=1)
        ax.legend(['before', 'after'])
        return fig,
    # animation function.  This is called sequentially
    def animate(i):
        print(i)
        ax.view_init(elev=10, azim=(i*30)%360)
        return fig,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=12, interval=3000, blit=True)

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    anim.save('animation.gif', writer='ffmpeg', fps=1)"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #color_map = plt.get_cmap('PiYG')
    color_map = plt.get_cmap('cividis')
    def init():
        ax.scatter(points_before[:, 0], points_before[:, 1], points_before[:, 2], c=points_before[:,3],cmap=color_map, s=0.02, alpha=0.1)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=0.02, alpha=1)
        ax.legend(['before', 'after'])
        return fig,

    def animate(i):
        print(i)
        ax.view_init(elev=10, azim=(i * 30) % 360)
        return fig,
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=12, interval=3000, blit=True)
    anim.save('animation.gif', writer='ffmpeg', fps=1)


def visualize_points_with_feature(points:np.ndarray,feature:np.ndarray, vis_num_tresh = 500000):

    if points.shape[0] > vis_num_tresh:
        print('too many points to visualize')
        return

    if points.shape[0] != feature.shape[0]:
        print("shapes of points and features not matching")
        return

    figure = plt.figure(1, bgcolor=(1, 1, 1), size=(640, 480))
    nodes = plt.points3d(points[:,0], points[:,1], points[:,2], feature, scale_factor=0.05, scale_mode='none',colormap='jet')
    colorbar = plt.colorbar()
    colorbar.label_text_property.color = (0, 0, 0)
    plt.show()

def visualize_points_with_normals(points:np.ndarray,normals:np.ndarray):
    if points.shape[0] != normals.shape[0]:
        print("pointcloud shape not matching with the normals shape")
        return

    figure = plt.figure(1,bgcolor=(1,1,1),size=(640,480))
    nodes = plt.points3d(points[:,0],points[:,1],points[:,2],points[:,3],scale_factor=0.05, scale_mode='none',colormap='jet')
    n = plt.quiver3d(points[:,0], points[:,1], points[:,2], normals[:,0], normals[:,1], normals[:,2],scale_factor=0.2, scale_mode='none',colormap='jet')
    plt.show()