# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import scipy.ndimage as ndi
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt

from google_static_maps_api import GoogleStaticMapsAPI
from google_static_maps_api import MAPTYPE
from google_static_maps_api import MAX_SIZE
from google_static_maps_api import SCALE


BLANK_THRESH = 2 * 1e-3     # Value below which point in a heatmap should be blank


def background_and_pixels(latitudes, longitudes, size, maptype):
    """Queries the proper background map and translate geo coordinated into pixel locations on this map.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param int size: target size of the map, in pixels
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: map and pixels
    :rtype: (PIL.Image, pandas.DataFrame)
    """
    # From lat/long to pixels, zoom and position in the tile
    center_lat = (latitudes.max() + latitudes.min()) / 2
    center_long = (longitudes.max() + longitudes.min()) / 2
    zoom = GoogleStaticMapsAPI.get_zoom(latitudes, longitudes, size, SCALE)
    pixels = GoogleStaticMapsAPI.to_tile_coordinates(latitudes, longitudes, center_lat, center_long, zoom, size, SCALE)
    # Google Map
    img = GoogleStaticMapsAPI.map(
        center=(center_lat, center_long),
        zoom=zoom,
        scale=SCALE,
        size=(size, size),
        maptype=maptype,
    )
    return img, pixels


def scatter(latitudes, longitudes, colors=None, maptype=MAPTYPE):
    """Scatter plot over a map. Can be used to visualize clusters by providing the marker colors.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series colors: marker colors, as integers
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    width = SCALE * MAX_SIZE
    colors = pd.Series(0, index=latitudes.index) if colors is None else colors
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))                                               # Background map
    plt.scatter(                                                            # Scatter plot
        pixels['x_pixel'],
        pixels['y_pixel'],
        c=colors,
        s=width / 20,
        linewidth=0,
        alpha=0.5,
    )
    plt.gca().invert_yaxis()                                                # Origin of map is upper left
    plt.axis([0, width, width, 0])                                          # Remove margin
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_markers(markers, maptype=MAPTYPE):
    """Plot markers on a map.

    :param pandas.DataFrame markers: DataFrame with at least 'latitude' and 'longitude' columnns, and optionally
        * 'color' column, see GoogleStaticMapsAPI docs for more info
        * 'label' column, see GoogleStaticMapsAPI docs for more info
        * 'size' column, see GoogleStaticMapsAPI docs for more info
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    img = GoogleStaticMapsAPI.map(scale=SCALE, markers=markers.T.to_dict().values(), maptype=maptype)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def heatmap(latitudes, longitudes, values, resolution=None, maptype=MAPTYPE):
    """Plot a geographical heatmap of the given metric.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series values: series of sample values
    :param int resolution: resolution (in pixels) for the heatmap
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)
    # Smooth metric
    z = grid_density_gaussian_filter(
        zip(pixels['x_pixel'], pixels['y_pixel'], values),
        MAX_SIZE * SCALE,
        resolution=resolution if resolution else MAX_SIZE * SCALE,          # Heuristic for pretty plots
    )
    # Plot
    width = SCALE * MAX_SIZE
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))                                               # Background map
    plt.imshow(z, origin='lower', extent=[0, width, 0, width], alpha=0.15)  # Foreground, transparent heatmap
    plt.scatter(pixels['x_pixel'], pixels['y_pixel'], s=1)                  # Markers of all points
    plt.gca().invert_yaxis()                                                # Origin of map is upper left
    plt.axis([0, width, width, 0])                                          # Remove margin
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def density_plot(latitudes, longitudes, resolution=None, maptype=MAPTYPE):
    """Given a set of geo coordinates, draw a density plot on a map.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param int resolution: resolution (in pixels) for the heatmap
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    heatmap(latitudes, longitudes, np.ones(latitudes.shape[0]), resolution=resolution, maptype=maptype)


def grid_density_gaussian_filter(data, size, resolution=None, smoothing_window=None):
    """Smoothing grid values with a Gaussian filter.

    :param [(float, float, float)] data: list of 3-dimensional grid coordinates
    :param int size: grid size
    :param int resolution: desired grid resolution
    :param int smoothing_window: size of the gaussian kernels for smoothing

    :return: smoothed grid values
    :rtype: numpy.ndarray
    """
    resolution = resolution if resolution else size
    k = (resolution - 1) / size
    w = smoothing_window if smoothing_window else int(0.01 * resolution)    # Heuristic
    imgw = (resolution + 2 * w)
    img = np.zeros((imgw, imgw))
    for x, y, z in data:
        ix = int(x * k) + w
        iy = int(y * k) + w
        if 0 <= ix < imgw and 0 <= iy < imgw:
            img[iy][ix] += z
    z = ndi.gaussian_filter(img, (w, w))                                    # Gaussian convolution
    z[z <= BLANK_THRESH] = np.nan                                           # Making low values blank
    return z[w:-w, w:-w]


def polygons(latitudes, longitudes, clusters, maptype=MAPTYPE):
    """Plot clusters of points on map, including them in a polygon defining their convex hull.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series clusters: marker clusters, as integers
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    width = SCALE * MAX_SIZE
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)

    polygons = []
    for c in clusters.unique():
        in_polygon = clusters == c
        if in_polygon.sum() < 3:
            print('[WARN] Cannot draw polygon for cluster {} - only {} samples.'.format(i, in_polygon.sum()))
            continue
        cluster_pixels = pixels.loc[clusters == c]
        polygons.append(Polygon(cluster_pixels.iloc[ConvexHull(cluster_pixels).vertices], closed=True))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plt.imshow(np.array(img))                                               # Background map
    p = PatchCollection(polygons, cmap='jet', alpha=0.15)                   # Collection of polygons
    p.set_array(clusters.unique())
    ax.add_collection(p)
    plt.scatter(                                                            # Scatter plot
        pixels['x_pixel'],
        pixels['y_pixel'],
        c=clusters,
        s=width / 40,
        linewidth=0,
        alpha=0.25,
    )
    plt.gca().invert_yaxis()                                                # Origin of map is upper left
    plt.axis([0, width, width, 0])                                          # Remove margin
    plt.axis('off')
    plt.tight_layout()
    plt.show()
