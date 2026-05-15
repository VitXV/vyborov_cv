import cv2
import numpy as np
from skimage.morphology import closing, opening
from skimage.measure import label, regionprops
import pyautogui
import time
import keyboard
import mss

template = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)

LetsGo = False
isStart = False
isBird = False
isJump = False
isDown = False

k = 1
updated_time = time.time()
last_jump_time = time.time()

with mss.MSS() as sct:
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    jump_start_time = time.time()
    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        now = time.time()
        if now - updated_time >= 10.0:
            k += 0.16

            updated_time = now

        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if keyboard.is_pressed('q'):
            break
        if keyboard.is_pressed('s'):
            isStart = True
        if keyboard.is_pressed('r'):
            k=1
            isStart = False
            LetsGo = False
            isBird = False
            isJump = False
            isDown = False
            last_press_time = 0

        if not(LetsGo):
            result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

        if isStart:
            gy1 = top_left[1]-92
            gy2 = gy1 + 92 + 96
            gx1 = top_left[0]
            gx2 = gx1 + 720

            mask1 = np.ones((3,3))
            mask2 = np.ones((6,6))

            isStart = False
            LetsGo = True

        if LetsGo:
            game = img[gy1:gy2, gx1:gx2].copy()
            game[0:32, 360:-1] = game[0,0]

            game = closing(game, mask1)
            game = opening(game, mask2)

            labeled = label(game)

            props = regionprops(labeled)
            sorted_props = sorted(props, key=lambda p: p.bbox[1])

            remap = np.arange(labeled.max() + 1)
            for new_id, prop in enumerate(sorted_props, start=1):
                remap[prop.label] = new_id
            labeled = remap[labeled]

            if len(sorted_props) > 2:
                y1 = sorted_props[1].bbox[0] 
                x1 = sorted_props[1].bbox[1]
                y2 = sorted_props[2].bbox[0]
                x2 = sorted_props[2].bbox[1]

                distance = abs(x2-x1)

                if distance < 150*k and 45<y2<87 and not isJump:
                    pyautogui.keyDown('down')
                    isBird = True

                if isBird and not isJump:
                    if not(distance < 150*k and 45<y2<87):
                        pyautogui.keyUp('down')
                        isBird=False
                        continue

                if distance < 120*k and not isJump:
                    pyautogui.press('up')
                    isJump = True
                    isDown = False
                    jump_start_time = now

                if isJump:
                    if 280 < sorted_props[2].area < 290 and not isDown:
                        if distance < 45 * k:
                            pyautogui.keyDown('down')
                            isDown = True
                            duck_start_time = now
                    if isDown:
                        passed_cactus = distance < 35
                        timeout = (now - duck_start_time) > 0.15
                        if passed_cactus or timeout:
                            pyautogui.keyUp('down')
                            isDown = False
                            isJump = False
                    if distance <= 25:
                        isJump = False
                        isDown = False
                        pyautogui.keyUp('down')
                if isJump and (now - jump_start_time) > 0.75 / k:
                    isJump = False
                    isDown = False
                    pyautogui.keyUp('down')
            #labeled = cv2.normalize(labeled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            #cv2.imshow("OpenCV/game", labeled)

