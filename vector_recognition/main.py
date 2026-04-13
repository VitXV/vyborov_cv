import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label,regionprops
from pathlib import Path

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] +2))
    new_image[1: -1, 1:-1] = region.image
    new_image = new_image * -1 +1
    labeled = label(new_image)
    return np.max(labeled)

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

def extractor(region):
    holes = count_holes(region)
    vlines,hlines = count_lines(region)
    vlines /= region.image.shape[0]
    hlines /= region.image.shape[0]
    eccentricity = region.eccentricity
    v_sym = symmetry(region)
    h_sym = symmetry(region,True)

    return np.array([region.area/region.image.size,holes,vlines,hlines,eccentricity,v_sym,h_sym])

def classificator(region,templates):
    features = extractor(region)
    res=""
    min_d = 10**6
    for symbol,t in templates.items():
        d = ((t-features)**2).sum() **0.5
        if d<min_d:
            res = symbol
            min_d = d
    return res

image = imread("alphabet-small.png")[:,:,:-1]
template = image.sum(2)
binary = template != 765.

labeled = label(binary)
props = regionprops(labeled)

templates = {}
for r, symbol in zip(props, ["8","0","A","B","1","W","X","*","/","-"]):
    templates[symbol] = extractor(r)

#print(classificator(props[0],templates))

image = imread("alphabet.png")[:,:,:-1]
abinary = image.mean(2)>0
alabeled = label(abinary)

aprops=regionprops(alabeled)

result = {}
save_path = Path(__file__).parent
image_path = save_path / "out"
image_path.mkdir(exist_ok=True)
plt.ion()
plt.figure(figsize=(5,7))
for region in aprops:
    symbol = classificator(region,templates)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] +=1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")
print()
print(result)
plt.imshow(props[0].image,cmap='flag')
plt.show()
