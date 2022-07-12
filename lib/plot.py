import numpy as np
import matplotlib.pyplot as plt


def plot_staticflow(image, dense, filename):
    cmap = "jet"
    fig, axes = plt.subplots(2, 1)
    titles = ["inputs",
              "staticFF"]

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow(image, cmap=cmap)
    ax2.imshow(dense, cmap=cmap)
    ax1.title.set_text(titles[0])
    ax2.title.set_text(titles[1])

    ax1.axis("off")
    ax2.axis("off")

    fig.savefig(filename, dpi=150)
