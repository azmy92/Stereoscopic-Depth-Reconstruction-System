__author__ = 'mostafa'

import cv2
import cv2.cv as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.mpl import axes
# Draws corresponding points
# I1: first image
# I2: second image
# matches: matcing points


def linearlstriangulation(u,P1,u1,P2):

    A = np.zeros((4, 3))

    A[0][0] = u[0]*P1[2][0]-P1[0][0]
    A[0][1] = u[0]*P1[2][1]-P1[0][1]
    A[0][2] = u[0]*P1[2][2]-P1[0][2]
    A[1][0] = u[1]*P1[2][0]-P1[1][0]
    A[1][1] = u[1]*P1[2][1]-P1[1][1]
    A[1][2] = u[1]*P1[2][2]-P1[1][2]
    A[2][0] = u1[0]*P2[2][0]-P2[0][0]
    A[2][1] = u1[0]*P2[2][1]-P2[0][1]
    A[2][2] = u1[0]*P2[2][2]-P2[0][2]
    A[3][0] = u1[1]*P2[2][0]-P2[1][0]
    A[3][1] = u1[1]*P2[2][1]-P2[1][1]
    A[3][2] = u1[1]*P2[2][2]-P2[1][2]



    B = np.zeros((4, 1))
    B[0][0] = -(u[0]*P1[2][3]-P1[0][3])
    B[1][0] = -(u[1]*P1[2][3]-P1[1][3])
    B[2][0] = -(u1[0]*P2[2][3]-P2[0][3])
    B[3][0] = -(u1[1]*P2[2][3]-P2[1][3])
    X = np.zeros((4, 1))
    AA = cv2.solve(A, B, X, cv2.DECOMP_SVD)
    print "========================cv2 solve========================================="
    print cv2.solve(A, B, X, cv2.DECOMP_SVD)[1]
    print "========================np linalg solve==================================="
    print np.linalg.lstsq(A, B)[0]

    return cv2.solve(A, B, X, cv2.DECOMP_SVD)[1]

def visualize(I1, I2, F, pt1, pt2):

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    #lines1 = cv2.computeCorrespondEpilines(pt2.reshape(-1, 1, 2), 1, F)
    #lines1 = lines1.reshape(-1, 3)
    #img1, img2 = drawlines(I1, I2, lines1, pt1, pt2)
    #cv2.imshow("Like In Assignment PDF", img1)
    #cv2.imshow("Like In Assignment PDF_2", img2)
    #cv2.waitKey(0)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(I2, I1, lines2, pt2, pt1)
    cv2.imshow("Like In Assignment PDF", img3)
    cv2.imshow("Like In Assignment PDF_2", img4)
    cv2.waitKey(0)
    #cv2.imwrite("E:\\4th Year\\Computer Vision\\ass5\\Report\\checker-Epilines.png",img3)


def drawlines(img1, img2, lines, pts1, pts2):
    r,c = img1.shape[:2]
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = (0, 0, 255)
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1, tuple(pt1), 2, color, -1)
        cv2.circle(img2, tuple(pt2), 2, color, -1)
    return img1, img2


def getfundamentals(matches,img1,img2):
    pt1 = np.zeros((matches.size/4, 2), 'float32')
    pt2 = np.zeros((matches.size/4, 2), 'float32')
    i = 0
    for m in matches:
        pt1[i] = (float(m[0]), float(m[1]))
        pt2[i] = (float(m[2]), float(m[3]))
        i += 1
    F, mask = cv2.findFundamentalMat(pt1, pt2, cv2.FM_LMEDS)


    return F, pt1, pt2


def drawcorrespondence(I1, I2, matches):
    img = np.array(np.hstack((I1, I2)))
    thickness = 2
    for m in matches:
    # draw the keypoints
        pt1 = (int(m[0]), int(m[1]))
        pt2 = (int(m[2] + I1.shape[1]), int(m[3]))
        lineColor = cv.CV_RGB(255, 0, 0)
        ptColor = cv.CV_RGB(0, 255, 0)
        cv2.circle(img, pt1, 2, ptColor, thickness)
        cv2.line(img, pt1, pt2, lineColor, thickness)
        cv2.circle(img, pt2, 2, ptColor, thickness)
        cv2.imshow("Matches", img)


def run():
    # Load images and match files for the first example
    I1 = cv2.imread('imgs/checker1.jpg')
    I2 = cv2.imread('imgs/checker2.jpg')
    # Load matching points
    matches = np.loadtxt('checker_matches.txt')
    # Load projection matrix for both cameras
    P1 = np.loadtxt("checker1_camera.txt")
    P2 = np.loadtxt("checker2_camera.txt")
    # This is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image:
    # matches(i,3:4) is a corresponding point in the second image

    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment
    drawcorrespondence(I1, I2, matches)
    # Calculating the Fundamental Matrix goes here
    F, pt1, pt2 = getfundamentals(matches,I1, I2)
    # Visulaize Epipolar lines here
    visualize(I1, I2, F, pt1, pt2)
    # Triangluation code goes here...
    pts1 = np.zeros((2, matches.size/4))
    pts2 = np.zeros((2, matches.size/4))
    j = 0
    for i in pt1:
        pts1[0][j] = i[0]
        pts1[1][j] = i[1]
        pts2[0][j] = pt2[j][0]
        pts2[1][j] = pt2[j][1]
        j += 1


    x = np.zeros((matches.size/4))
    y = np.zeros((matches.size/4))
    z = np.zeros((matches.size/4))
    print x.size
    j = 0
    for i in pt1:
        p = linearlstriangulation(i, P1, pt2[j], P2)
        x[j] = p[0]
        y[j] = p[1]
        z[j] = p[2]
        j += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range (0,j):
        xs = x[i]
        ys = y[i]
        zs = z[i]
        ax.scatter(xs, ys, zs)

    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    #ax.set_xlim3d(x.min(), x.max())
    #ax.set_ylim3d(y.min(), y.max())
    #ax.set_zlim3d(z.min(), z.max())

    plt.show()


   # fig.savefig('E:\\4th Year\\Computer Vision\\ass5\\Report\\checker-Figure.png', dpi=fig.dpi)

    #X = cv2.triangulatePoints( P1[:3], P2[:3], pts1[:2], pts2[:2])
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.legend()



    while True:
        cv2.cv.WaitKey(0)

if __name__ == "__main__":
    run()
