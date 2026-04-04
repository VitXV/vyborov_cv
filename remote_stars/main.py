import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label
import socket

host = "84.237.21.36"
port = 5152

def recv_all(sock,nbytess):
    data = bytearray()
    while len(data) < nbytess:
        packet = sock.recv(nbytess-len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def calculate(image):
    x1,y1 = ndimage.center_of_mass(image==1)
    x2,y2 = ndimage.center_of_mass(image==2)
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

plt.ion()
plt.figure()
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((host,port))
    sock.send(b"124ras1")
    print(sock.recv(10))
    beat = b"nope"
    while beat != b"yep":
        sock.send(b"get")
        bts = recv_all(sock,40002)
        im1 = np.frombuffer(bts[2:40002], dtype="uint8")
        im1 = im1.reshape(bts[0],bts[1])
        objs = label(im1>160)
        sock.send(str((round(calculate(objs), 1))).encode())
        print(sock.recv(10))

        sock.send(b"beat")
        beat = sock.recv(10)

        plt.clf()
        plt.imshow(objs)
        plt.pause(1)
