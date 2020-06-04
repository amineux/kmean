
# coding: utf-8

# #Clustering

# ##Algorithm de  K-Means Algorithm

# Pratique k-means Clustering


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=300, centers=3,
                  random_state=0, cluster_std=0.60)
print X.shape, y.shape 

plt.scatter(X[:, 0], X[:, 1], s=30);

from sklearn.cluster import KMeans
ktest = KMeans(n_clusters=3)
ktest.fit(X) 
y_pred = ktest.predict(X) 

plt.scatter(X[:,0], X[:,1], c=y_pred)

ktest = KMeans(n_clusters=4) 
ktest.fit(X) 
y_pred = ktest.predict(X) 

plt.scatter(X[:,0], X[:,1], c=y_pred)

ktest = KMeans(n_clusters=3, max_iter=100) 
ktest.fit(X)
y_pred = ktest.predict(X) 

plt.scatter(X[:,0], X[:,1], c=y_pred)

X, y = make_blobs(n_samples=300, centers=3,
                  random_state=0, cluster_std=0.85)
print X.shape, y.shape 
# Visualisation
plt.scatter(X[:, 0], X[:, 1], s=30);

ktest = KMeans(n_clusters=3, max_iter=100) 
ktest.fit(X) 
y_pred = ktest.predict(X)
plt.scatter(X[:,0], X[:,1], c=y_pred)

fig, ax=plt.subplots(1,2,figsize=(8,4))
ax[0].scatter(X[:,0], X[:,1],c=y)
ax[1].scatter(X[:,0], X[:,1],c=y_pred)
from sklearn.datasets import load_digits
digits = load_digits()
print digits.keys()
print digits.data.shape
print digits.target
X,y=digits.data, digits.target
k_digits = KMeans(n_clusters=10)
y_pred=k_digits.fit_predict(X)

print k_digits.cluster_centers_.shape

fig = plt.figure(figsize=(8,3))
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.reshape(k_digits.cluster_centers_[i],(8,8)), cmap=plt.cm.binary)

from sklearn.decomposition import RandomizedPCA
pca=RandomizedPCA(2).fit(X)
X_proj = pca.transform(X)

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].scatter(X_proj[:,0], X_proj[:,1], c=y_pred)
ax[0].set_title('Clusters reduced to 2D with PCA', fontsize=10)

ax[1].scatter(X_proj[:,0], X_proj[:,1], c=y)
ax[1].set_title('Original Dataset reduced to 2D with PCA', fontsize=10)



from sklearn.datasets import load_sample_image
img=load_sample_image("china.jpg");
plt.imshow(img)



print img.shape


img_r = (img / 255.0).reshape(-1,3)
print img_r.shape



k_colors = KMeans(n_clusters=64).fit(img_r)
y_pred=k_colors.predict(img_r)

ster centers. We must have a total of 64 centroids, shape must be of the input 

print k_colors.cluster_centers_.shape 


print k_colors.labels_.shape 


newimg=k_colors.cluster_centers_[k_colors.labels_]
print newimg.shape 
newimg=np.reshape(newimg, (img.shape))
print newimg.shape 

ax=fig.add_subplot(1,2,1,xticks=[],yticks=[],title='Original Image')
ax.imshow(img)
ax=fig.add_subplot(1,2,2,xticks=[],yticks=[],title='Color Compressed Image using K-Means')
ax.imshow(newimg)
plt.show()
