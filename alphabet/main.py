import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from pathlib import Path

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] +2))
    new_image[1: -1, 1:-1] = region.image
    new_image = new_image * -1 +1
    labeled = label(new_image)
    return np.max(labeled) -1

def count_lines(region):
    shape = region.image.shape
    image = region.image
    vlines = (np.sum(image,0) / shape[0] == 1).sum()
    hlines = (np.sum(image,1) / shape[1] == 1).sum()
    return vlines, hlines

def symmetry(region, transpose=False):
    image = region.image
    if transpose:
        image=image.T
    shape = image.shape
    top = image[:shape[0]//2]
    if image.shape[0]%2==0:
        bottom = image[shape[0]//2:]
    else:
        bottom = image[shape[0]//2+1:]
    bottom = bottom[::-1]
    result = top == bottom
    return result.sum() / result.size

def classificator(region):
    holes = count_holes(region)
    if holes == 2: #B, 8
        v, _ = count_lines(region)
        v /= region.image.shape[1]

        if v > 0.2:
            return "B"
        else:
            return "8"
    elif holes == 1: #A, O, P, D
        h_sym = symmetry(region)
        v_sym = symmetry(region, transpose=True)

        if h_sym > 0.80 and v_sym > 0.80:
            return "0"
        elif h_sym > 0.9 and v_sym > 0.5:
            return "D"
        elif h_sym > 0.25 and v_sym > 0.75:
            return "A"
        else:
            return "P"
    elif holes == 0: #1, W, X, *, -. /
        h_sym = symmetry(region)
        v_sym = symmetry(region, transpose=True)

        if h_sym > 0.999 and v_sym > 0.999:
            return "-"
        elif h_sym > 0.75 and v_sym > 0.75:
            if region.eccentricity > 0.85:
                return "1"
            else:
                return "X"
        elif h_sym > 0.25 and v_sym > 0.75:
            if region.eccentricity > 0.5:
                return "W"
            else:
                return "*"
        else:
            return "/"
    return "?"


image = imread('symbols.png')
binary = image.mean(2) > 0
labeled = label(binary)

props = regionprops(labeled)

save_path = Path(__file__).parent
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.ion()
plt.figure(figsize=(5,7))

result = {}
for r in props:
    symbol = classificator(r)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] +=1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(r.image)
    plt.savefig(image_path / f"image_{r.label}.png")

print('\n', result)
plt.imshow(props[0].image,cmap='flag')
plt.show()
