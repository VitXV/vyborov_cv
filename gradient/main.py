import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")

color1 = [255, 0, 128]
color2 = [0, 128, 255]

for i, v in enumerate(np.linspace(0, 1, size)):
    for j, w in enumerate(np.linspace(0, 1, size)):
        x=(v+w)/2
        r=lerp(color1[0],color2[0],x)
        g=lerp(color1[1],color2[1],x)
        b=lerp(color1[2],color2[2],x)
        image[i][j] = [r,g,b]

plt.imshow(image)
plt.show()