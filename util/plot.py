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

def plot_skeleton(connection, pos_t, posconf_t = None,
                    dotsize=50, c=['g', 'r', 'b'], alpha=0.8,
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
    connection: list of tuple
        Connect relationship list of the lines between joints. 
    pos_t: float (len(idx_pos),)
        Absolute position array of joints. 
    posconf_t: float (num_pos,)
        Position-confidence array of joints.

    dotsize: int
        Plotting size of each joints. 
    c: list
        Color of confident joints, unconfident joints and lines. 
    alpha: float
        Transparency of plotting. 

    ax:
        Panel to plot. 
    view: list of int (2)
        Camera view as [elev, azim].
        'elev' stores the elevation angle in the z plane. 
        'azim' stores the azimuth angle in the x,y plane.
    """
    if ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        if xlim != None or ylim != None or zlim != None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    if view != None:
        ax.view_init(view[0], view[1])

    posX = [pos_t[i*3+0] for i in range(len(pos_t)/3)]
    posY = [pos_t[i*3+1] for i in range(len(pos_t)/3)]
    posZ = [pos_t[i*3+2] for i in range(len(pos_t)/3)]

    uncfposX = [pos_t[i*3+0] for i in find(posconf_t == 0)]
    uncfposY = [pos_t[i*3+1] for i in find(posconf_t == 0)]
    uncfposZ = [pos_t[i*3+2] for i in find(posconf_t == 0)]

    ax.scatter(posX, posY, posZ, c=c[0], s=dotsize, alpha=alpha)
    ax.scatter(uncfposX, uncfposY, uncfposZ, c=c[1], s=dotsize, alpha=alpha)

    lineX1 = [pos_t[p1*3+0] for (p1, p2) in connection]
    lineY1 = [pos_t[p1*3+1] for (p1, p2) in connection]
    lineZ1 = [pos_t[p1*3+2] for (p1, p2) in connection]

    lineX2 = [pos_t[p2*3+0] for (p1, p2) in connection]
    lineY2 = [pos_t[p2*3+1] for (p1, p2) in connection]
    lineZ2 = [pos_t[p2*3+2] for (p1, p2) in connection]

    for i in range(len(connection)):
        ax.plot([lineX1[i], lineX2[i]], 
                [lineY1[i], lineY2[i]], 
                [lineZ1[i], lineZ2[i]], c[2])

    return

def animate_skeleton(connection, pos, posconf=None, 
                        dotsize=50, linesize=30, c=['g', 'r', 'b'], alpha=0.8,
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
    connection: list of tuple
        Connect relationship list of the lines between joints. 
    pos: float (len_seq, len(idx_pos))
        Absolute position array of joints. 
    posconf: float (len_seq, num_pos)
        Position-confidence array of joints.

    dotsize: int
        Plotting size of each joints. 
    c: list
        Color of confident joints, unconfident joints and lines. 
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
    # create fig and ax with default size if not given
    if fig == None or ax == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        if xlim != None or ylim != None or zlim != None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)

    if view != None:
        ax.view_init(view[0], view[1])

    len_seq = pos.shape[0]

    # initialize
    plot_pos =\
        ax.scatter([], [], [], c=c[0], s=dotsize, alpha=alpha, animated=True)
    plot_uncfpos =\
        ax.scatter([], [], [], c=c[1], s=dotsize, alpha=alpha)

    plot_lines =\
            [ax.plot([], [], [], c[2]) for i in range(len(connection))]

    def init():
        plot_pos._offsets3d = (np.ma.ravel([]), 
                                    np.ma.ravel([]), 
                                    np.ma.ravel([]))

        plot_uncfpos._offsets3d = (np.ma.ravel([]), 
                                    np.ma.ravel([]), 
                                    np.ma.ravel([]))

        for line in plot_lines:
            line[0].set_data([], [])
            line[0].set_3d_properties([])

        return [plot_pos] + [plot_uncfpos] + plot_lines

    def update(t):
        pos_t = pos[t, :]
        if posconf == None:
            posconf_t = None
        else:
            posconf_t = posconf[t, :]

        posX = [pos_t[i*3+0] for i in range(len(pos_t)/3)]
        posY = [pos_t[i*3+1] for i in range(len(pos_t)/3)]
        posZ = [pos_t[i*3+2] for i in range(len(pos_t)/3)]

        uncfposX = [pos_t[i*3+0] for i in find(posconf_t == 0)]
        uncfposY = [pos_t[i*3+1] for i in find(posconf_t == 0)]
        uncfposZ = [pos_t[i*3+2] for i in find(posconf_t == 0)]

        lineX1 = [pos_t[p1*3+0] for (p1, p2) in connection]
        lineY1 = [pos_t[p1*3+1] for (p1, p2) in connection]
        lineZ1 = [pos_t[p1*3+2] for (p1, p2) in connection]
        
        lineX2 = [pos_t[p2*3+0] for (p1, p2) in connection]
        lineY2 = [pos_t[p2*3+1] for (p1, p2) in connection]
        lineZ2 = [pos_t[p2*3+2] for (p1, p2) in connection]

        plot_pos._offsets3d = (np.ma.ravel(posX), 
                                    np.ma.ravel(posY), 
                                    np.ma.ravel(posZ))
        plot_uncfpos._offsets3d = (np.ma.ravel(uncfposX), 
                                    np.ma.ravel(uncfposY), 
                                    np.ma.ravel(uncfposZ))

        i = 0
        for line in plot_lines:
            line[0].set_data([lineX1[i], lineX2[i]], [lineY1[i], lineY2[i]])
            line[0].set_3d_properties([lineZ1[i], lineZ2[i]])
            i += 1
        # TODO: display frame number on the plot
        return [plot_pos] + [plot_uncfpos] + plot_lines

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