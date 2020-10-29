import cv2
import numpy as np


def getVideosTop():
    fl = cv2.VideoCapture('vids/fr_Dec17.mp4')
    bot = cv2.VideoCapture('vids/bot_all_paws_Jan24.mp4')
    fr = cv2.VideoCapture('vids/fr_Dec17.mp4')
    bl = cv2.VideoCapture('vids/fr_Dec17.mp4')
    br = cv2.VideoCapture('vids/back_right_Jan23.mp4')

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
