import cv2
import numpy as np
from PyQt5 import QtCore


def getVideosTop():
    fl = cv2.VideoCapture('drop.avi')
    bot = cv2.VideoCapture('drop.avi')
    fr = cv2.VideoCapture('drop.avi')
    bl = cv2.VideoCapture('drop.avi')
    br = cv2.VideoCapture('drop.avi')

    print(fl.get(cv2.CAP_PROP_FPS))
    print(fl.get(cv2.CAP_PROP_FRAME_COUNT))
    videos = [fl, bot, fr, bl, br]

    if not fl.isOpened():
        print("error")

    while fl.isOpened():
        ret_fl, frame_fl = fl.read()
        ret_bot, frame_bot = bot.read()
        ret_fr, frame_fr = fr.read()
        ret_bl, frame_bl = bl.read()
        ret_br, frame_br = br.read()
        blank_image = np.zeros(shape=frame_fl.shape, dtype=np.uint8)
        if ret_fl:
            top = np.concatenate((frame_fl, frame_bot, frame_fr), axis=1)
            bottom = np.concatenate((frame_bl, blank_image, frame_br), axis=1)
            final = np.concatenate((top, bottom), axis=0)
            cv2.imshow("Frame", final)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    for vid in videos:
        vid.release()

    cv2.destroyAllWindows()
    return final


if __name__ == '__main__':
    g = getVideosTop()
