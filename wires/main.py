import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening

image = np.load('wires/wires1.npy')
labeled = label(image)

struct = np.ones((3, 1))
cut = opening(labeled, struct)

for i in range(1,labeled.max()+1):
    wire = cut==i
    parts = label(wire).max()
    print(parts)

plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(cut)
plt.show()