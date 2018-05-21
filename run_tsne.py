import sklearn
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import imageio
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.mlab as mlab
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans

# trying to compare the tsne with pca
# will also compare runtimes

with_images = False

def load_data(directory):
    arr = []

    ans = None
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            temp_filename = directory + filename
            arr.append(temp_filename)
            print(temp_filename)
            new_image = imageio.imread(temp_filename).flatten()
            print(new_image.shape)
            if ans is None:
                ans = new_image
            else:
                ans = np.vstack((ans, new_image))

    return ans, arr

def plot_images(embeddings, arr, plot_images=False, save_name=None, show_image=True, pred=None, centers=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # colors = cm.Spectral(np.linspace(0, 1, num_classes))
        xx = embeddings[:, 0]
        yy = embeddings[:, 1]

        # plot the images
        if plot_images == True:
            for i, (x, y) in enumerate(zip(xx, yy)):
                print("CREATING IMAGES FOR " + str(i))
                im = OffsetImage(imageio.imread(arr[i]), zoom=0.03)
                im.set_height(3)
                im.set_width(3)
                ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
                ax.add_artist(ab)
            ax.update_datalim(np.column_stack([xx, yy]))
            ax.autoscale()

        # plot the 2D data points
        for i in range(embeddings.shape[0]):
            print("PLOTTING " + str(i))
            ax.scatter(xx[i], yy[i], c=pred, s=50)

        if centers:
            ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)



        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
        plt.legend(loc='best', scatterpoints=1, fontsize=5)
        if save_name is not None:
            plt.savefig(save_name, format='png', dpi=600)
        if show_image:
            plt.show()

# def k_means(embeddings):




tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
data, arr = load_data("test/")
embeddings = tsne.fit_transform(data, arr)

# plot the initial embeddings
plot_images(embeddings, arr, plot_images=True, save_name="plots/pre_clustering_plot_0.03_no_images.png", show_image=False)

# k-means example
kmeans = KMeans(n_clusters=4)
kmeans.fit(embeddings)
y_kmeans = kmeans.predict(embeddings)
centers = kmeans.cluster_centers_
plot_images(embeddings, arr, plot_images=True, save_name="plots/pre_clustering_plot_kmeans_4_images.png", show_image=True, pred=y_kmeans, centers=centers)


