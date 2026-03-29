import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label

images=[]
for i in range(100):
    image = np.load(f'out/h_{i}.npy')
    images.append(image)

def distance(x1,y1,x2,y2):
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def centers(image):
    labeled = label(image)
    mx=np.max(labeled)
    centers=[]
    for i in range(1,mx+1):
        obj = labeled==i
        centers.append(ndimage.center_of_mass(obj))
    return centers

track = [[c] for c in centers(images[0])]
for i in range(1,100):
    cntrs = centers(images[i])
    for t in track:
        last = t[-1]
        dx = np.argmin([distance(last[0],last[1],c[0],c[1]) for c in cntrs])
        t.append(cntrs[dx])

plt.figure()
for t in track:
    ys, xs = zip(*t)
    plt.plot(xs, ys,marker='.')

plt.show()
