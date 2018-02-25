
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import time
from scipy import misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='input the path to image', type=str)
parser.add_argument('-n', '--numberOfCluster', nargs='?', const = 5, help='number of clusters for colors', type=int)
parser.add_argument('-d', '--dest', help='destination to store result image', type=str)
parser.add_argument('-s', '--show', action='store_true', help='display the output image')
args = parser.parse_args()

#load image in and set it to variable aurora
inputImg = misc.imread(args.path)

#convert aurora from dtype unit8 to float64 and normalize it by division
img = np.array(inputImg, dtype=np.float64) /255
#obtain dimensions of input image
w,h,d = img.shape
print('The shape of your input image is height {}, width {}, channel {}'.format(w,h,d))

#reshape aurora into dimension (height*width, 3)
img = np.reshape(img, [-1, 3])

#set number of colors classification
n_colors = args.numberOfCluster
#perform k-mean
t0 = time()
kmeans = KMeans(n_clusters= n_colors, n_init=10, random_state=0).fit(img)
print('Completed the k-mean modeling in %.2f seconds' % round(time()-t0,2) )
print('Starting to create new image...')
print('='*31)

t0 = time()
#color segmentation
color_labels = kmeans.cluster_centers_
#recreate images by setting each individual pixel to its according color label
pixel_label = kmeans.fit_predict(img)
#recreate the image with only cluster colors
output_img = [color_labels[i] * 255 for i in pixel_label]
#reshape the output_image_array into image format
output_img = np.reshape(output_img, [w,h,d])
print('Completed recreating new image in %0.2f' % round(time()-t0))

if args.dest:
    #saving output image to destination path
    misc.imsave(args.dest, output_img)
    print('new image is store in', args.dest)

if args.show:
    #shwo the output_image
    plt.imshow(output_img)
    plt.show()
