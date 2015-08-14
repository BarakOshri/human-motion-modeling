# Note: This is a Z-up coordinatfe system

# This line configures matplotlib to show figures embedded in the notebook, 
# instead of opening a new window for each figure. More about that later. 
# If you are using an old version of IPython, try using '%pylab inline' instead.

# The below line is used in iPython
# %matplotlib inline

from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

from IPython.display import HTML

def plot_skeleton(skel, 
                    pos_t, pos_joi_t=None,
                    dotsize=40, alpha=0.8,
                    ax=None, view=None,
                    figsize=(10, 10), 
                    xlim=(-1500, 1500), 
                    ylim=(-1500, 1500), 
                    zlim=(-1500, 1500)):
    """
    Plot a single frame of the skeleton. 
    (The coordainate system has a z-up frame. ) 

    Parameters
    ----------
    skel: 
        Skeleton structure of the MoCap data.
    pos_t: float array
        World position of joints. 
    pos_joi_t: float array
        World position of joints of interest.

    dotsize: int
        Plotting size of each joints. 
    alpha: float
        Transparency of plotting. 

    ax:
        Panel to plot. 
    view: list of int (2)
        Camera view as [elev, azim].
        'elev' stores the elevation angle in the z plane. 
        'azim' stores the azimuth angle in the x,y plane.
    """
    # construct the figure if not given
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        if xlim != None or ylim != None or zlim != None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    if view != None:
        ax.view_init(view[0], view[1])

    connection = skel['connection']

    posX = [pos_t[i*3+0] for i in range(len(pos_t)/3)]
    posY = [pos_t[i*3+1] for i in range(len(pos_t)/3)]
    posZ = [pos_t[i*3+2] for i in range(len(pos_t)/3)]
    ax.scatter(posX, posY, posZ, c='g', s=dotsize, alpha=alpha) 
    # plot joints of interests
    if pos_joi_t != None:
        pos_joiX = [pos_joi_t[i*3+0] for i in range(len(pos_joi_t)/3)]
        pos_joiY = [pos_joi_t[i*3+1] for i in range(len(pos_joi_t)/3)]
        pos_joiZ = [pos_joi_t[i*3+2] for i in range(len(pos_joi_t)/3)]
        ax.scatter(pos_joiX, pos_joiY, pos_joiZ, c='y', s=dotsize, alpha=alpha)

    # plot lines
    lineX1 = [pos_t[p1*3+0] for (p1, p2) in connection]
    lineY1 = [pos_t[p1*3+1] for (p1, p2) in connection]
    lineZ1 = [pos_t[p1*3+2] for (p1, p2) in connection]

    lineX2 = [pos_t[p2*3+0] for (p1, p2) in connection]
    lineY2 = [pos_t[p2*3+1] for (p1, p2) in connection]
    lineZ2 = [pos_t[p2*3+2] for (p1, p2) in connection]

    for i in range(len(connection)):
        ax.plot([lineX1[i], lineX2[i]], 
                [lineY1[i], lineY2[i]], 
                [lineZ1[i], lineZ2[i]], 'b')

    return


def animate_skeleton(skel, 
                        pos_arr, 
                        pos_joi_arr=None,
                        n_seed=0,
                        dotsize=40, alpha=0.8,
                        fig=None, ax=None, view=None,
                        figsize=(10, 10), 
                        xlim=(-1500, 1500), 
                        ylim=(-1500, 1500), 
                        zlim=(-1500, 1500)):
    """
    Make an animation of the skeleton movements. 
    Switch (x, y, z) to (x, z, y) for better visualization.

    Parameters
    ----------
    skel: 
        Skeleton structure of the MoCap data.
    pos_arr: float (len_seq, len(idx_pos))
        World position of joints. 
    pos_joi_arr: float array
        World position of joints of interest.

    dotsize: int
        Plotting size of each joints. 
    alpha: float
        Transparency of plotting. 

    fig: 
        Panel figure to plot. 
    ax:
        Panel axis to plot. 
    view: list of int (2)
        Camera view as [elev, azim].
        'elev' stores the elevation angle in the z plane. 
        'azim' stores the azimuth angle in the x,y plane.

    figsize: int tuple
        Figure size. 
    xlim: int tuple
        Limitation of the x axis on the plot. 
    ylim: int tuple
        Limitation of the y axis on the plot. 
    zlim: int tuple
        Limitation of the z axis on the plot. 
    """
    # construct the figure if not given
    if fig == None or ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        if xlim != None or ylim != None or zlim != None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    if view != None:
        ax.view_init(view[0], view[1])

    connection = skel['connection']

    len_seq = pos_arr.shape[0]

    plot_frame_num =\
        ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    plot_pos =\
        ax.scatter([], [], [], c='g', s=dotsize, alpha=alpha, animated=True)
    plot_pos_joi =\
        ax.scatter([], [], [], c='y', s=dotsize, alpha=alpha)

    plot_lines =\
            [ax.plot([], [], [], 'b') for i in range(len(connection))]

    def init():
        plot_frame_num.set_text('')

        plot_pos._offsets3d = (np.ma.ravel([]), 
                                    np.ma.ravel([]), 
                                    np.ma.ravel([]))

        for line in plot_lines:
            line[0].set_data([], [])
            line[0].set_3d_properties([])

        return [plot_pos] + [plot_pos_joi] +plot_lines

    def update(t):
        # frame number
        plot_frame_num.set_text('frame: {}'.format(t))

        # skeleton joints
        pos_t = pos_arr[t, :]
        posX = [pos_t[i*3+0] for i in range(len(pos_t)/3)]
        posY = [pos_t[i*3+1] for i in range(len(pos_t)/3)]
        posZ = [pos_t[i*3+2] for i in range(len(pos_t)/3)]

        pos_joiX = []
        pos_joiY = []
        pos_joiZ = []
        # joints of interest
        if pos_joi_arr == None:
            pos_joi_t = None
        else:
            pos_joi_t = pos_joi_arr[t, :]
            pos_joiX = [pos_joi_t[i*3+0] for i in range(len(pos_joi_t)/3)]
            pos_joiY = [pos_joi_t[i*3+1] for i in range(len(pos_joi_t)/3)]
            pos_joiZ = [pos_joi_t[i*3+2] for i in range(len(pos_joi_t)/3)]

        # lines
        lineX1 = [pos_t[p1*3+0] for (p1, p2) in connection]
        lineY1 = [pos_t[p1*3+1] for (p1, p2) in connection]
        lineZ1 = [pos_t[p1*3+2] for (p1, p2) in connection]
        
        lineX2 = [pos_t[p2*3+0] for (p1, p2) in connection]
        lineY2 = [pos_t[p2*3+1] for (p1, p2) in connection]
        lineZ2 = [pos_t[p2*3+2] for (p1, p2) in connection]

        # update the plot
        plot_pos._offsets3d = (np.ma.ravel(posX), 
                                    np.ma.ravel(posY), 
                                    np.ma.ravel(posZ))
        plot_pos_joi._offsets3d = (np.ma.ravel(pos_joiX), 
                                    np.ma.ravel(pos_joiY), 
                                    np.ma.ravel(pos_joiZ))

        if t < n_seed:
            line_color = 'g'
        else:
            line_color = 'b'

        i = 0
        for line in plot_lines:
            line[0].set_data([lineX1[i], lineX2[i]], [lineY1[i], lineY2[i]])
            line[0].set_3d_properties([lineZ1[i], lineZ2[i]])
            line[0].set_color(line_color)
            i += 1

        return [plot_pos] + [plot_pos_joi] + plot_lines

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len_seq, interval=1, blit=True)
    return anim


def play_animation(filename='animation.mp4'):
    """
    Play the animation file (.mp4) on iPython notebook via HTML5. 
    # Uility function of animation play in iPython
    # Note: Not supported well by Chrome
    """
    video = open(filename, "rb").read()
    video_encoded = video.encode("base64")
    video_tag = '<video controls alt="test" src="data:video/x-m4v;base64,{0}">'\
                .format(video_encoded)
    HTML(video_tag)
