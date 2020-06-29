import matplotlib.pyplot as plt


def plot_rn():
    """
    3d plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def plot_actions(ax, vertices, all_vertices, n_v, actions):
    """
    Plot states after applying action in latent space
    """
    actioned_z = [all_vertices[n_v*(i+1):n_v*(i+2)]
                  for i in range(actions.shape[0])]

    for a, l in enumerate(actioned_z):
        for i, v in enumerate(vertices):
            lin = l[i]
            ax.plot([lin[0], v[0]], [lin[1], v[1]], [lin[2], v[2]],
                    c="grey")
    return ax


def plot_vertices(vertices, ax=None, fig=None, colors=None, bar=True,
                  bar_label="Values"):
    """
    Plot states + values
    """
    try:
        z_dim = vertices[:, 2]
    except IndexError:
        z_dim = 0

    if colors is None:
        colors = ["black" for i in range(vertices.shape[0])]
        bar = False
        vmin= -1
        vmax= 1
    else:
        vmin = min(colors)
        vmax = max(colors)

    im = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    c=colors, s=30, zorder=200, edgecolors="black",
                    vmin=vmin, vmax=vmax)
    if bar:
        cbar = fig.colorbar(im)
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(bar_label, rotation=270)

    return ax, fig

