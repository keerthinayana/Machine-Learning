from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import winsound
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    vs = cv2.VideoCapture(args["video"])
    initial_w = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(initial_w)
    initial_h = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(initial_h)
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    print(fps)
    frame_count = 1
    print(frame_count)
    # Subtract the background
    fgbg = cv2.createBackgroundSubtractorMOG2()
   # feature_params = dict(maxCorners=100, qualitylevel=0.3, minDistance=7, blockSize=7)
   # lk_params = dict(winSize=(15, 15), maxlevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))

    ret, old_frame = vs.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
  #  p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    firstFrame = None
    totalFrames = 0

    while True:
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Detecting"
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame is None:
            break
        frame_count = frame_count + 1
        # Change the size of the frame
        frame = imutils.resize(frame, width=500)
        # Initialize the count of the people
        people_count = 0
        totalDown = 0
        totalUp = 0
        # Converting to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Remove the background
        fgbgmask = fgbg.apply(gray)

        # Find the direction
        #p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        #good_new = p1[st == 1]
        #good_old = p0[st == 1]

        if firstFrame is None:
            firstFrame = gray
            continue

        frameDelta = cv2.absdiff(firstFrame, gray)
        # Threshold the image
        thresh = cv2.threshold(fgbgmask, 25, 255, cv2.THRESH_BINARY) [1]
        dilated = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = "Movement"
            # Increment the count of the people
        people_count = people_count + 1
        totalUp = totalUp + 1
        totalDown = totalDown + 1
        people_count_message = "People Count : " + str(people_count)
        up_msg = "Up :" + str(totalUp)
        down_msg = "Down :" + str(totalDown)
        cv2.putText(frame, "Status:{}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame, up_msg, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(frame, down_msg, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        cv2.putText(frame, people_count_message, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1)
        # Give a beep if the people count is more than the limit
        if people_count > 6:
            frequency = 2500  # Set Frequency To 2500 Hertz
            duration = 1000  # Set Duration To 1000 ms == 1 second
            winsound.Beep(frequency, duration)
            # Display the frames
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", dilated)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        totalFrames += 1
       # old_gray = frame_gray.copy()
    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()






