import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("data/project/train/L1000957.JPG", cv2.IMREAD_GRAYSCALE)  # queryImage
img2 = cv2.imread("data/project/references/Passion_au_lait.JPG", cv2.IMREAD_GRAYSCALE)  # trainImage

image_height, image_width = 675, 900
object_centers = {
    "Amandina": (2440, 3160),
    "Arabia": (2500, 3311),
    "Comtesse": (2122, 3247),
    "Creme_brulee": (2261, 3382),
    "Jelly_Black": (2249, 3330),
    "Jelly_Milk": (2208, 3355),
    "Jelly_White": (2238, 3310),
    "Noblesse": (2050, 3431),
    "Noir_authentique": (2148, 3296),
    "Passion_au_lait": (2055, 3217),
    "Stracciatella": (2335, 3674),
    "Tentation_noir": (2290, 3232),
    "Triangolo": (2017, 3240),
}
center_i, center_j = object_centers["Passion_au_lait"]
i0 = center_i - image_height // 2
i1 = center_i + (image_height + 1) // 2
j0 = center_j - image_width // 2
j1 = center_j + (image_width + 1) // 2
img2 = img2[i0:i1, j0:j1]

img1 = cv2.resize(img1, dsize=None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, dsize=None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=100)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3)
plt.show()
