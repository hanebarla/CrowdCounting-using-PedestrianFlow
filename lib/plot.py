import numpy as np
import matplotlib.pyplot as plt


def plot_staticflow(image, flow, dense, filename):
    cmap = "jet"
    fig, axes = plt.subplots(3, 4)
    titles = ["inputs",
              "dense",
              "up left",
              "up",
              "up right",
              "left",
              "stay",
              "right",
              "down left",
              "down",
              "down right",
              "potential"]

    for i in range(12):
        if i == 0:
            axes[i//4][i%4].imshow(image)
        elif i == 1:
            axes[i//4][i%4].imshow(dense, cmap=cmap)
        else:
            axes[i//4][i%4].imshow(flow[i-2, ...], cmap=cmap)

        axes[i//4][i%4].title.set_text(titles[i])
        axes[i//4][i%4].axis("off")

    fig.savefig(filename, dpi=150)
