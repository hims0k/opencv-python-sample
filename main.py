import cv2
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def print_hi(name):
    # read image
    img = cv2.imread('./src/image/grapes.jpg')
    # change to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # change to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('gray', img_gray)
    cv2.imshow('hsv', img_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'{img_gray.shape}')


def printing(x):
    print(x)
    initial_value = 100
    cv2.namedWindow('img')
    # create track bar
    cv2.createTrackbar('track', 'img', initial_value, 255, printing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)


def print_hi2():
    img = cv2.imread('src/image/grapes.jpg')
    cv2.namedWindow('image')
    # configure mouse setting
    cv2.setMouseCallback('image', print_position)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_hi3():
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # draw line
    cv2.line(img, (0, 0), (150, 190), (255, 0, 0), 2)
    # draw rectangle
    cv2.rectangle(img, (100, 25), (300, 150), (0, 255, 0), 3)
    # draw circle
    cv2.circle(img, (100, 100), 55, (0, 0, 255), -1)
    # draw ellipse
    cv2.ellipse(img, (250, 250), (100, 50), 20, 180, 360, (255, 0, 0), 1)

    # configure font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # put text
    cv2.putText(img, 'OpenCV', (100, 300), font, 1, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gamma_convert():
    gamma = 0.5
    img = cv2.imread('src/image/Berry.jpg')
    # numpy array for convert
    gamma_cvt = np.zeros((256, 1), dtype='uint8')

    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i) / 255) ** (1.0 / gamma)
    # convert gamma
    img_gamma = cv2.LUT(img, gamma_cvt)

    cv2.imshow('img', img)
    cv2.imshow('gamma', img_gamma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_equalization():
    plt.style.use('ggplot')

    # change to Japanese font
    mpl.rcParams['font.family'] = 'IPAGothic'

    img = cv2.imread('src/image/lenna.jpg', 0)
    # calc Histogram
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # img = plt.plot(hist)
    # plt.xlabel('画素数')
    # plt.ylabel('頻度')
    # plt.show()

    img_eq = cv2.equalizeHist(img)
    hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
    cv2.imshow('img', img)
    cv2.imshow('img_hist', img_eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binarization():
    img = cv2.imread('src/image/grapes.jpg', 0)
    cv2.namedWindow('img')
    threshold = 100
    cv2.createTrackbar('track', 'img', threshold, 255, on_trackbar)

    while True:
        ret, img_th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        # substitute threshold value
        threshold = cv2.getTrackbarPos('track', 'img')
        cv2.imshow('img', img_th)

        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()


def on_trackbar(position):
    pass


def affine_transform_para():
    img = cv2.imread('src/image/grapes.jpg')
    h, w = img.shape[:2]
    # 平行移動料
    dx, dy = 30, 30
    # 変換行列の作成
    afn_mat = np.float32([[1, 0, dx], [0, 1, dy]])
    # execute affine transform
    img_afn = cv2.warpAffine(img, afn_mat, (w, h))
    cv2.imshow('affine', img_afn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def affine_transform_rotate():
    img = cv2.imread('src/image/grapes.jpg')
    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w/2, h/2), 40, 1)
    # execute affine transform
    img_afn = cv2.warpAffine(img, rot_mat, (w, h))

    cv2.imshow('img_rot', img_afn)
    cv2.imwrite('img_affine2.png', img_afn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def morphology_transform():
    img = cv2.imread('src/image/floor.jpg')
    ret, img_th = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    cv2.imshow('img_th', img_th)
    cv2.imshow('img', img)

    # prepare necessary kernel to calc morphology
    kernel = np.ones((3, 3), dtype=np.uint8)
    # dilate
    img_d = cv2.dilate(img_th, kernel)
    # erode
    img_e = cv2.erode(img_th, kernel)
    # closing
    img_c = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)
    cv2.imshow('dilate', img_d)
    cv2.imshow('erode', img_e)
    cv2.imshow('closing', img_c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def smoothing():
    img = cv2.imread('src/image/Lena.jpg')
    # select position to generate noise
    x = np.random.randint(512, size=500)
    y = np.random.randint(512, size=500)
    # insert noise
    for i, j in zip(x, y):
        img[i, j] = 150

    # define smoothing kernel
    average_kernel = np.ones((3, 3)) / 9.0
    # convoluted average kernel
    img_ave = cv2.filter2D(img, -1, average_kernel)

    cv2.imshow('img_average', img_ave)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def smoothing_fn():
    img = cv2.imread('src/image/Lena.jpg')
    # select position to generate noise
    x = np.random.randint(512, size=500)
    y = np.random.randint(512, size=500)
    # insert noise
    for i, j in zip(x, y):
        img[i, j] = 150

    # execute convolution process
    img_blur = cv2.blur(img, (3, 3))
    img_ga = cv2.GaussianBlur(img, (9, 9), 2)
    img_me = cv2.medianBlur(img, 5)
    img_bi = cv2.bilateralFilter(img, 20, 30, 30)

    cv2.imshow('img_blur', img_blur)
    cv2.imshow('img_ga', img_ga)
    cv2.imshow('img_me', img_me)
    cv2.imshow('img_bi', img_bi)
    cv2.imshow('img_orig', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_edge():
    img = cv2.imread('src/image/Lena.jpg', 0)
    # sobel filter
    img_sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    img_sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    # change data-type to uint8 of differential image data
    img_sobelx = cv2.convertScaleAbs(img_sobelx)
    img_sobely = cv2.convertScaleAbs(img_sobely)

    cv2.imshow('x', img_sobelx)
    cv2.imshow('y', img_sobely)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_edge_laplacian():
    img = cv2.imread('src/image/Lena.jpg')

    # ラプラシアンフィルタ
    img_lap = cv2.Laplacian(img, cv2.CV_32F)
    # convert to uint8
    img_lap = cv2.convertScaleAbs(img_lap)
    cv2.imshow('img_lap', img_lap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_edge_canny():
    img = cv2.imread('src/image/Lena.jpg')
    # Canny
    img_canny = cv2.Canny(img, 100, 100)
    cv2.imshow('canny', img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_line():
    img = cv2.imread('src/image/road.jpg')
    # Canny edge detection
    img_g = cv2.imread('src/image/road.jpg', 0)
    img_canny = cv2.Canny(img_g, 300, 400)
    cv2.imshow('img', img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # detect line
    lines = cv2.HoughLines(img_canny, 1, np.pi / 180, 100)

    # draw detected lines
    for i in lines[:]:
        rho = i[0][0]
        theta = i[0][1]
        cos = np.cos(theta)
        sin = np.sin(theta)
        x0 = cos * rho
        y0 = sin * rho
        x1 = int(x0 + 1000 * (-sin))
        y1 = int(y0 + 1000 * cos)
        x2 = int(x0 - 1000 * (-sin))
        y2 = int(y0 - 1000 * cos)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_circle():
    img = cv2.imread("src/image/grapes.jpg", 0)
    # 円の検出
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=20, param2=35, minRadius=1,
                               maxRadius=30)

    img = cv2.imread("src/image/grapes.jpg")
    #
    # 検出した円を重ねる
    #
    for i in circles[0]:
        cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 1)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def harris():
    img = cv2.imread('src/image/buildings.jpg')
    img_g = cv2.imread('src/image/buildings.jpg', 0)

    img_harris = copy.deepcopy(img)
    # detect feature of Harris
    img_dst = cv2.cornerHarris(img_g, 2, 3, 0, 0.04)
    print(img_dst)
    # draw feature point
    img_harris[img_dst > 0.05 * img_dst.max()] = [0, 0, 255]
    cv2.imshow('img', img_harris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def kaze():
    img = cv2.imread('src/image/buildings.jpg')
    img_g = cv2.imread('src/image/buildings.jpg', 0)

    img_akaze = copy.deepcopy(img)
    # prepare AKAZE
    akaze = cv2.AKAZE_create()
    kp1 = akaze.detect(img_akaze)
    img_akaze = cv2.drawKeypoints(img_akaze, kp1, None)

    img_orb = copy.deepcopy(img)
    # prepare ORB
    orb = cv2.ORB_create()
    kp2 = orb.detect(img_orb)
    img_orb = cv2.drawKeypoints(img_orb, kp2, None)

    cv2.imshow('AKAZE', img_akaze)
    cv2.imshow('ORB', img_orb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_face():
    # HaarLike 特徴のパス
    HAAR_FILE = 'C:/ProgramData/Anaconda3/envs/img-recog-sample/Lib/site-packages/cv2/data/haarcascade_profileface.xml'

    cascade = cv2.CascadeClassifier(HAAR_FILE)

    img = cv2.imread('src/image/shibuya_kousaten.png')
    face = cascade.detectMultiScale(img)
    print(face)
    # draw rectangle of face line
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_contour():
    img = cv2.imread('src/image/shibuya_kousaten.png')
    img_g = cv2.imread('src/image/shibuya_kousaten.png', 0)

    # binarization
    ret, img_bi = cv2.threshold(img_g, 20, 255, cv2.THRESH_BINARY)
    # extract contour
    contours, hierarchy = cv2.findContours(img_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    print(hierarchy)
    # draw contours
    img_contour = cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    cv2.imshow('img', img_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_blob():
    img = cv2.imread('src/image/Blob.png')
    img_g = cv2.imread('src/image/Blob.png', 0)
    # binarization
    ret, img_bi = cv2.threshold(img_g, 100, 255, cv2.THRESH_BINARY)
    # detect blob
    n_labels, label_image, stats, centroids = cv2.connectedComponentsWithStats(img_bi)
    # print(n_labels)
    print(stats)
    # print(centroids)

    # draw features on image
    img_blob = copy.deepcopy(img)
    h, w = img_g.shape
    print(img_g.shape)
    color = [255, 0, 0]
    # paint blob
    for y in range(h):
        for x in range(w):
            if label_image[y, x] > 0:
                img_blob[y, x] = color

    for i in range(1, n_labels):
        xc = int(centroids[i][0])
        yc = int(centroids[i][1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (255, 255, 255)
        cv2.putText(img_blob, str(stats[i][-1]), (xc, yc), font, scale, color)

    cv2.imshow('img', img_blob)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_color_of_movie():
    cap = cv2.VideoCapture('src/movie/Mobility.mp4')
    while True:
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 640, 480)
        ret, frame = cap.read()
        if not ret:
            break
        # transform to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 50, 50])
        upper = np.array([25, 255, 255])
        # create mask to extract specified color
        frame_mask = cv2.inRange(hsv, lower, upper)
        # detect color
        dst = cv2.bitwise_and(frame, frame, mask=frame_mask)
        cv2.imshow('img', dst)
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()


def optical_flow():
    # 抜き出す特徴店の数
    count = 500
    # 特徴点を探すときの収束条件
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.03)
    # Lucas-Kanade に用いるパラメータ
    lk_params = dict(winSize=(10, 10), maxLevel=4, criteria=criteria)
    cap = cv2.VideoCapture('src/movie/Cosmos.mp4')
    _, frame = cap.read()
    frame_pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        _, frame = cap.read()
        if not _:
            break
        frame_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 追うべき特徴点を探す
        feature_pre = cv2.goodFeaturesToTrack(frame_pre, count, 0.001, 5)
        if feature_pre is None:
            continue
        # オプティカルフロー
        feature_now, status, err = cv2.calcOpticalFlowPyrLK(frame_pre, frame_now, feature_pre, None, **lk_params)
        for i in range(len(feature_now)):
            pre_x = feature_pre[i][0][0]
            pre_y = feature_pre[i][0][1]
            now_x = feature_now[i][0][0]
            now_y = feature_now[i][0][1]
            cv2.line(frame, (int(pre_x), int(pre_y)), (int(now_x), int(now_y)), (255, 0, 0), 3)

        cv2.imshow('img', frame)
        frame_pre = frame_now.copy()

        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()


def mean_shift_cam_shift():
    cap = cv2.VideoCapture('src/movie/Cruse.mp4')
    _, frame = cap.read()
    h, w, ch = frame.shape

    # 探索窓の初期位置、大きさ
    rct = (600, 500, 200, 200)
    # MeanShiftの収束条件
    cri = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
    while True:
        th = 100
        _, frame = cap.read()
        if not _:
            break
        img_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(img_g, th, 255, cv2.THRESH_BINARY)
        # MeanShift
        # _, rct = cv2.meanShift(img_bin, rct, cri)
        # CamShift
        _, rct = cv2.CamShift(img_bin, rct, cri)
        x, y, w, h = rct
        # 探索窓を四角形で表示
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.imshow('img', frame)
        if cv2.waitKey(10) == 27:
            break

    cv2.destroyAllWindows()


def diff_bg():
    cap = cv2.VideoCapture('src/movie/People.mp4')
    _, frame = cap.read()
    h, w, ch = frame.shape

    # 背景差分用の背景
    frame_back = np.zeros((h, w, ch), dtype=np.float32)
    while True:
        _, frame = cap.read()
        if not _:
            break
        # 現在のフレームとの背景とで差分を取る
        frame_diff = cv2.absdiff(frame.astype(np.float32), frame_back)
        # 背景を少しずつ現在のフレームに近づけたものを新たな背景とする
        cv2.accumulateWeighted(frame, frame_back, 0.03)
        cv2.imshow('img', frame_diff.astype(np.uint8))
        if cv2.waitKey(10) == 27:
            break
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    diff_bg()
