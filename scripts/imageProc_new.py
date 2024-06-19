#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import scipy
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import itertools

from torch import zero_
from cv_bridge import CvBridge
import math
from copy import deepcopy

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from contours import *
from geometric import *
from featureMatching import *
from movingVariance import *

class imageProc:
    def __init__(self, scannerParams, contourParams, roiParams, trackerParams):
        """# class FeatureExtractor to extract the the line Features: bottom point, angle

        Args:
            windowProp (_type_): _description_
            roiParams (_type_): _description_
            featureParams (_type_): _description_
        """
        self.roiParams = roiParams
        self.contourParams = contourParams
        self.scannerParams = scannerParams
        self.trackerParams = trackerParams

        self.reset()

    def reset(self):
        print("**************Reset*************")
        self.count = 0
        self.pointsInTop = 0
        self.pointsInBottom = 0

        # parameters
        self.bushy = False
        self.CropRows = []
        self.primaryRGBImg = []
        self.cropLaneFound = False
        self.isInitialized = False

        self.cropRowEnd = False

        self.trackingBoxLoc = []

        self.buffersize = 30
        # features
        self.allLineEnd = []
        self.allLineStart = []
        self.mainLine_up = np.array([])
        self.mainLine_down = np.array([])
        
        # window parameters
        self.trackingWindowOffsTop = 0
        self.trackingWindowOffsBottom = 0
        self.trackingWindowTopScaleRatio = 0.4  # scales the top of the window

        # self.imgHeight, self.imgWidth = 720, 1280# realsense->480, 640, oakd->720, 1280
        self.imgHeight, self.imgWidth = 480, 640
        self.depth_intrinsics = None  

        self.line_ends_metric = None
        self.plants_metric = None   

        # steps create linspace
        self.scanFootSteps = np.linspace(self.scannerParams["scanStartPoint"],
                                         self.scannerParams["scanEndPoint"],
                                         self.scannerParams["scanWindowWidth"])
        self.rowTrackingBoxes = []
        self.updateTrackingBoxes()

        self.numOfCropRows = len(self.rowTrackingBoxes)
        self.bufferzone = np.zeros((20, 2))

    def findCropLane(self, rgbImg, depthImg, camera_params, mode='RGB-D', bushy=False):
        """finds Crops Rows in the image based on RGB and depth data
        Args:
            bushy (bool, optional): _description_. Defaults to False.
        """
        self.primaryRGBImg = rgbImg.copy()
        self.primaryDepthImg = depthImg.copy()
        self.depth_intrinsics = camera_params
        # plt.imshow(self.primaryDepthImg)
        # plt.show()
        self.imgHeight, self.imgWidth, self.imgChannels = rgbImg.shape
        # if crop canopy type is busshy like Coriander
        self.bushy = bushy
        
        # check if current image is not Empty or None
        if self.primaryRGBImg is None or len(self.primaryRGBImg) == 0:
            print("[bold red]#[ERR][/] CR-Scanner - Image is None, Check Topics or the Sensor !")
        else:
            # initial RGB image process to get mask GreenIDx and plant centers
            self.mask, self.greenIDX, self.plantObjects2D, self.plantCenters2D = self.processRGBImage(self.primaryRGBImg)
            self.numPlantsInScene = len(self.plantCenters2D[0])            
        if not self.isInitialized:
            print("[bold blue]#[INF][/] Find Crop Lane")
            self.lines2D, self.linesROIs2D = self.findCropRows2D(self.primaryRGBImg)
            # self.lines3D, self.linesROIs3D = self.findCropRows3D(self.mask, self.plantCenters2D, self.primaryDepthImg)
            # merge both predictions to get more robust results!

        return self.cropLaneFound

    def findCropRows3D(self, maskImg, plantCenters2D, depthImg):
        lines = []
        linesROIs = []
        # project the plant centers to 3D

        # cluster points 

        # fit line on each cluster

        # project back 3D points to image

        # return the resutls
        return lines, linesROIs

    def findCropRows2D(self, rgbImg):
        # search for lines at given window locations using LQ-Adjustment
        # print("running findcrop2d")
        lines, linesROIs, _ = self.findLinesInImage()
        # print("lines", len(lines))
        # if any feature is observed during the scan
        if len(lines) != 0:
            np.set_printoptions(suppress=True)
            # compute moving standard deviation
            mvSignal = movingStd(lines[:, 0])
            # find positive an negative peaks of the signal
            peaksPos, peaksNeg = findPicksTroughths(mvSignal, 0.5)
            # locate best lines
            self.CropRows, self.trackingBoxLoc = self.findCropRowsInMVSignal(peaksPos, peaksNeg,
                                                        mvSignal, lines, linesROIs)
            self.numOfCropRows = len(self.trackingBoxLoc)
            print("numofcroprow", self.numOfCropRows)
            self.lostCropRows = list(np.zeros(self.numOfCropRows))
            # if there is any proper line to follow
            if self.numOfCropRows != 0:
                # set parameters indicating a successfull initialization
                self.isInitialized = True
                self.cropLaneFound = True
                print('[bold blue]#[INF][/] Controller Initialized - Crop Rows:',
                        len(self.CropRows),
                        ', Window positions:',
                        self.CropRows[:,0].tolist())
            else:
                print(
                    '--- Initialisation failed - No lines detected due to peak detection ---')
                self.cropLaneFound = False
        else:
            print(
                '--- Initialisation failed - No lines detected by sweeping window ---')
            self.cropLaneFound = False
        return lines, linesROIs

    def findCropRowsInMVSignal(self, peaksPos, peaksNeg, mvSignal, lines, linesROIs):
        # if multiple lines
        # print("runing this function")
        if len(peaksPos) != 0 and len(peaksNeg) != 0 and len(mvSignal) != 0:
            # best line one each side of the crop row transition
            qualifiedLines = np.zeros((len(peaksPos)+1, 2))
            windowLocations = np.zeros((len(peaksPos)+1, 1))
            # get lines at each side of a positive peak (= crop row
            # transition and select the best)
            try:
                lidx = 0
                # every peak stands for 2 croprows
                for k in range(0, len(peaksPos) + 1):
                    # first peak
                    if k == 0:
                        peaksNegLine = peaksNeg[peaksNeg < peaksPos[k]]
                    else:
                        # second to last-1
                        if k < len(peaksPos):
                            linestmp = peaksNeg[peaksNeg < peaksPos[k]]
                            peaksNegLine = linestmp[linestmp > peaksPos[k-1]]
                        # last peak
                        else:
                            peaksNegLine = peaksNeg[peaksNeg > peaksPos[k-1]]

                    if len(peaksNegLine) != 0:
                        # best line (measured by variance)
                        bestLine = np.where(
                            mvSignal[peaksNegLine] == np.min(mvSignal[peaksNegLine]))[0][0]
                        # parameters of that line
                        qualifiedLines[lidx, :] = lines[peaksNegLine[bestLine]]
                        # location of the window which led to this line
                        if linesROIs[peaksNegLine[bestLine]] != [0]:
                            windowLocations[lidx,
                                                 :] = linesROIs[peaksNegLine[bestLine]]
                            lidx += 1
            except:

                # fallback: just take all negative peaks
                qualifiedLines = lines[peaksNeg]
                windowLocations = np.mean(linesROIs[peaksNeg])

        # if there are no positive peaks but negative ones: no crop
        # row transition -> there might be just one line
        elif len(peaksPos) == 0 and len(peaksNeg) != 0 and len(mvSignal) != 0:
            peaksNegLine = peaksNeg
            bestLine = np.where(mvSignal[peaksNegLine] == np.min(mvSignal[peaksNegLine]))[0][0]

            qualifiedLines = np.zeros((1, 2))
            windowLocations = np.zeros((1, 1))

            qualifiedLines[0, :] = lines[peaksNegLine[bestLine]]
            windowLocations[0] = linesROIs[peaksNegLine[bestLine]]
        else:
            qualifiedLines = []
            windowLocations = []
        
        qualifiedLines = qualifiedLines[~np.all(qualifiedLines == 0, axis=1)]
        # if len(qualifiedLines) > 0:
        #     qualifiedLines = qualifiedLines[np.any(qualifiedLines != 0, axis=1)]

        windowLocations = windowLocations[~np.all(windowLocations == 0, axis=1)]
        # if len(windowLocations) > 0:
        #     windowLocations = windowLocations[np.any(windowLocations != 0, axis=1)]
        return qualifiedLines,  windowLocations

    def findLinesInImage(self):
        """function to get lines at given window locations 
        Args:
            self.scanFootSteps (_type_): _description_
        Returns:
            _type_: lines in image
        """
        # initialization
        lines = np.zeros((len(self.scanFootSteps), 2))
        trackingWindows = np.zeros((len(self.scanFootSteps), 1))
        meanLinesInWindows = np.zeros((len(self.scanFootSteps), 1))
        angleVariance = None
        
        # counting the points in the top and bottom half
        self.pointsInBottom = 0
        self.pointsInTop = 0
        
        # for all windows
        # print("num of crop rows", self.numOfCropRows)
        for boxIdx in range(0, self.numOfCropRows, 1):
            # define window
            if self.isInitialized:
                # get x values, where the top and bottom of the window intersect
                # with the previous found line
                lineIntersection = lineIntersectWin(self.CropRows[boxIdx, 1],
                                                    self.CropRows[boxIdx, 0],
                                                    self.imgHeight,
                                                    self.trackerParams["topOffset"],
                                                    self.trackerParams["bottomOffset"])
                self.updateTrackingBoxes(boxIdx, lineIntersection)

            plantsInCropRow = []
            for ptIdx in range(self.numPlantsInScene):
                # checking #points in upper/lower half of the image
                self.checkPlantsLocTB(self.plantCenters2D[:, ptIdx])
                
                # if plant center is inside tracking box
                if self.rowTrackingBoxes[boxIdx].contains(Point(self.plantCenters2D[0, ptIdx], self.plantCenters2D[1, ptIdx])):
                    plantsInCropRow.append(self.plantCenters2D[:, ptIdx])

            if len(plantsInCropRow) >= 2:
                # flipped points
                # print(plantsInCropRow)
                ptsFlip = np.flip(plantsInCropRow, axis=1)
                # print(ptsFlip)
                # get line at scanning window
                xM, xB = getLineRphi(ptsFlip)
                t_i, b_i = lineIntersectImgUpDown(xM, xB, self.imgHeight)
                l_i, r_i = lineIntersectImgSides(xM, xB, self.imgWidth)
                # print(xM, xB)
                # print("row ID:", boxIdx, t_i, b_i, l_i, r_i )
                # print("plantsInCropRow", np.array(plantsInCropRow))
                self.plants_metric = self.get_3d_points(np.array(plantsInCropRow))
                # if the linefit does not return None and the line-image intersections
                # are within the image bounds
                
                pixels = self.get_pixels(t_i=t_i, b_i=b_i)
                points_in_3d = []
                for i in range(pixels.shape[0]):
                    point_in_3d = self.get3dpoints(depthImg = self.primaryDepthImg, pixels = pixels[i])
                    points_in_3d.append(point_in_3d)
                # print("points in 3d", points_in_3d)


                if xM is not None and b_i >= 0 and b_i <= self.imgWidth:
                    lines[boxIdx, :] = [xB, xM]
                    # store the window location, which generated these line
                    # parameters
                    if self.isInitialized == False:
                        trackingWindows[boxIdx] = self.scanFootSteps[boxIdx]
                    # mean x value of all points used for fitting the line
                    meanLinesInWindows[boxIdx] = np.median(plantsInCropRow, axis=0)[0]

        # if the feature extractor is initalized, adjust the window
        # size with the variance of the line fit
        if self.isInitialized and angleVariance is not None:
            self.trackerParams["trackingBoxWidth"] = max(3*angleVariance, self.trackerParams["trackingBoxWidth"])

        # delete zero rows from line parameters
        lines = lines[~np.all(lines == 0, axis=1)]
        trackingWindows = trackingWindows[~np.all(trackingWindows == 0, axis=1)]
        meanLinesInWindows = meanLinesInWindows[~np.all(meanLinesInWindows == 0, axis=1)]
        if len(lines) < 10:
            if self.buffersize * len(lines) != self.bufferzone.shape[0]:
                self.bufferzone = np.zeros((self.buffersize * len(lines), 2))
        # print("lines numerber:", lines)
            for i in range(lines.shape[0]):
                new_line = self.bufferforlines(line= lines[i], lineID = i)
                lines[i] = new_line
        # print("lines:", lines)
        
        return lines, trackingWindows, meanLinesInWindows

    def updateTrackingBoxes(self, boxID=0, lineIntersection=None):

        if not self.isInitialized and lineIntersection==None:
            # initial tracking Boxes 
            for i in range(len(self.scanFootSteps)):
                # window is centered around self.scanFootSteps[i]
                boxBL_x = self.scanFootSteps[i]
                boxBR_x = self.scanFootSteps[i] + self.trackerParams["trackingBoxWidth"]
                boxTL_x = int(boxBL_x - self.trackerParams["trackingBoxWidth"]/2 * self.trackerParams["sacleRatio"])
                boxTR_x = int(boxTL_x + self.trackerParams["trackingBoxWidth"] * self.trackerParams["sacleRatio"])
                boxT_y = self.trackerParams["bottomOffset"]
                boxB_y = self.imgHeight - self.trackerParams["topOffset"]
                # store the corner points
                self.rowTrackingBoxes.append(Polygon([(boxBR_x, boxB_y),
                                                      (boxBL_x, boxB_y), 
                                                      (boxTL_x, boxT_y),
                                                      (boxTR_x, boxT_y)]))
        else:
            # window corner points are left and right of these
            boxBL_x = int(lineIntersection[0] - self.trackerParams["trackingBoxWidth"]/2)
            boxBR_x = int(boxBL_x + self.trackerParams["trackingBoxWidth"])
            boxTL_x = int(lineIntersection[1] - self.trackerParams["trackingBoxWidth"]/2 * self.trackerParams["sacleRatio"])
            boxTR_x = int(boxTL_x + self.trackerParams["trackingBoxWidth"] * self.trackerParams["sacleRatio"])
            boxT_y = self.trackerParams["bottomOffset"]
            boxB_y = self.imgHeight - self.trackerParams["topOffset"]
            # store the corner points
            self.rowTrackingBoxes[boxID] = Polygon([(boxBR_x, boxB_y),
                                                    (boxBL_x, boxB_y),
                                                    (boxTL_x, boxT_y),
                                                    (boxTR_x, boxT_y)])

    def checkPlantsInRows(self, cropRowID, plantsInCroRow):

        if len(plantsInCroRow) >= 2:
            return True
        else:
            self.lostCropRows[cropRowID] = 1
            return False   

    def checkPlantsLocTB(self, point):
        if point[1] < self.imgHeight/2:
            self.pointsInTop += 1
        else:
            self.pointsInBottom += 1

    def trackCropLane(self, mode=None):
        """function to get lines at given window locations 
        Returns:
            _type_: _description_
        """
        P, ang = None, None
        # if the feature extractor is initalized
        if self.cropLaneFound:
            # get the lines at windows defined through previous found lines
            lines, linesROIs, meanLinesInWindows = self.findLinesInImage()
            print("trackingboxloc", len(self.trackingBoxLoc))
            # if 'all' lines are found by 'self.getLinesInImage'
            if len(lines) >= len(self.trackingBoxLoc):
                print("len(lines) >=")
                # the line parameters are the new tracked lines (for the next step)
                self.CropRows = lines
                # location is always the left side of the window
                self.trackingBoxLoc = meanLinesInWindows - self.trackerParams["trackingBoxWidth"]/2
                # average image intersections of all found lines
                avgOfLines = np.mean(np.c_[lineIntersectImgUpDown(
                    self.CropRows[:, 1], self.CropRows[:, 0], self.imgHeight)], axis=0)
                #  get AvgLine in image cords
                self.mainLine_up, self.mainLine_down = getImgLineUpDown(
                    avgOfLines, self.imgHeight)
                # compute all intersections between the image and each line
                allLineIntersect = np.c_[lineIntersectWin(self.CropRows[:, 1],
                                                          self.CropRows[:, 0],
                                                          self.imgHeight,
                                                          self.trackerParams["topOffset"],
                                                          self.trackerParams["bottomOffset"])]
                # store start and end points of lines - mainly for plotting
                self.allLineStart = np.c_[
                    allLineIntersect[:, 0], self.imgHeight - self.trackerParams["bottomOffset"] * np.ones((len(self.CropRows), 1))]
                self.allLineEnd = np.c_[
                    allLineIntersect[:, 1], self.trackerParams["topOffset"] * np.ones((len(self.CropRows), 1))]
                
                line_ends = self.remove_duplicates(np.vstack((self.allLineStart, self.allLineEnd)))
                print("line_ends:", line_ends)
                self.line_ends_metric = self.get_3d_points(line_ends)

                #test
                # pixels = np.array([])

                print("self.allLineStart", self.allLineStart)
                print("self.allLineEnd", self.allLineEnd)
                print("metric_line_ends", self.line_ends_metric)
                # main features
                self.P = self.cameraToImage([avgOfLines[1], self.imgHeight])
                self.ang = computeTheta(self.mainLine_up, self.mainLine_down)
                self.cropLaneFound = True

            else:
                # print("line when false", len(lines))
                # print("trackingboxloc", len(self.trackingBoxLoc))
                print("[bold red]#[ERR][/] Lost at least one line")
                self.cropLaneFound = False
        else:
            print('Running rest()..')
        
        if self.pointsInBottom == 0 and self.pointsInTop == 0:
            self.cropLaneFound = False
        # print("croprowend:", self.pointsInTop)
        if self.pointsInBottom == 0 and mode in [2, 5]:
            self.cropRowEnd = True
        elif self.pointsInTop == 0 and mode in [1, 4]:
            self.cropRowEnd = True
        else:
            self.cropRowEnd = False
        
        return self.cropLaneFound, self.cropRowEnd, P, ang

    def processRGBImage(self, rgbImg):
        """function to extract the greenidx and the contour center points
        Returns:
            _type_: _description_
        """
        # change datatype to enable negative values for self.greenIDX calculation
        rgbImg = rgbImg.astype('int32')

        # cols_top = [self.imgWidth*4/13, self.imgWidth*9/13] #self.imgHeight, self.imgWidth
        # cols_bottom = [self.imgWidth*3/15, self.imgWidth*9/10]        #LEFT corner & RIGHT corner
        cols_top = [self.imgWidth*0, self.imgWidth*1] #self.imgHeight, self.imgWidth
        cols_bottom = [self.imgWidth*0, self.imgWidth*1]

        points_mask = np.array([[cols_top[0], 0], [cols_top[1], 0], [cols_bottom[1], self.imgHeight], [cols_bottom[0], self.imgHeight]])
        # mask = np.ones(rgbImg.shape, dtype=np.uint8)
        mask = np.zeros_like(rgbImg)
        cv.fillPoly(mask, pts=np.int32([points_mask]), color=(255, 255, 255))

        rgbImg = cv.bitwise_and(rgbImg, mask)

        # #LEFT corner & RIGHT corner
        # for i in range(3):
        #     for j in range(50,290):
        #         rgbImg[j, : 290 - j -50, i] = 0
        #         rgbImg[j, -1:j - 350:-1 ,i] = 0 
        # rgbImg[:50, :240, :3] = 0
        # rgbImg[:50, 370:640, :3] = 0
        
        #SPLIT MASK
        for i in range(40, self.imgHeight, 30):
            rgbImg[i: i+15, :self.imgWidth, :3] = 0

        # rgbImg[:720, 600:750, :3] = 0

        # apply ROI
        rgbImg = self.applyROI(rgbImg)
        #  get green index and binary mask (Otsu is used for extracting the green)
        mask, greenIDX = self.getExgMask(rgbImg)
        # plt.imshow(mask)
        # plt.show()
        # find contours
        plantObjects = getPlantMasks(
            mask, self.contourParams["minContourArea"], bushy=False)
        # get center of contours
        contCenterPTS = getContourCenter(plantObjects)
        x = contCenterPTS[:, 1]
        y = contCenterPTS[:, 0]
        # cv.imshow("RGB image",mask.astype('uint8'))
        # self.handleKey(0)
        return mask, greenIDX, plantObjects, np.array([x, y])

    def getExgMask(self, img):
        """Function to compute the green index of the image
        """
        _img = img.copy()

        _img = _img.astype('int32')
        # Vegetation Mask
        r = _img[:, :, 0]
        g = _img[:, :, 1]
        b = _img[:, :, 2]

        # calculate Excess Green Index and filter out negative values
        greenIDX = 2*g - r - b
        greenIDX[greenIDX < 0] = 0
        greenIDX = greenIDX.astype('uint8')

        # Otsu's thresholding after gaussian smoothing
        blur = cv.GaussianBlur(greenIDX, (5, 5), 0)
        threshold, threshIMG = cv.threshold(
            blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # dilation
        kernel = np.ones((10, 10), np.uint8)
        binaryMask = cv.dilate(threshIMG, kernel, iterations=1)

        # Erotion
        # er_kernel = np.ones((10,10),dtype=np.uint8) # this must be tuned
        # binaryMask= cv.erode(binaryMask, er_kernel)

        return binaryMask, greenIDX

    def applyROI(self, img):
        _img = img.copy()
        # defining ROI windown on the image
        if self.roiParams["enable_roi"]:
            r_pts = [self.roiParams["p1"], self.roiParams["p2"],
                     self.roiParams["p3"], self.roiParams["p4"]]
            l_pts = [self.roiParams["p5"], self.roiParams["p6"],
                     self.roiParams["p7"], self.roiParams["p8"]]

            cv.fillPoly(_img, np.array([r_pts]), (0, 0, 0))
            cv.fillPoly(_img, np.array([l_pts]), (0, 0, 0))
        return _img

    def cameraToImage(self, P):
        """function to transform the feature point from camera to image frame
        Args:
            P (_type_): point in camera Fr
        Returns:
            _type_: point in image Fr
        """
        P[0] = P[0] - self.imgWidth/2
        P[1] = P[1] - self.imgHeight/2
        return P

    def drawGraphics(self):
        """function to draw the lines and the windows onto the image (self.primaryRGBImg)
        """
        if self.primaryRGBImg != []:
            self.graphicsImg = self.primaryRGBImg.copy()
            # main line
            # print("MAIN LINE", int(self.mainLine_up[0]), int(self.mainLine_up[1]))
            cv.line(self.graphicsImg, (int(self.mainLine_up[0]), int(self.mainLine_up[1])), (int(
                self.mainLine_down[0]), int(self.mainLine_down[1])), (255, 0, 0), thickness=3)
            # contoures
            cv.drawContours(self.graphicsImg,
                            self.plantObjects2D, -1, (10, 50, 150), 3)
            for i in range(0, len(self.allLineStart)):
                # helper lines
                cv.line(self.graphicsImg, (int(self.allLineStart[i, 0]), int(self.allLineStart[i, 1])), (int(
                    self.allLineEnd[i, 0]), int(self.allLineEnd[i, 1])), (0, 255, 0), thickness=1)

            for i in range(self.numOfCropRows):
                int_coords = lambda x: np.array(x).round().astype(np.int32)
                exterior = [int_coords(self.rowTrackingBoxes[i].exterior.coords)]
                cv.polylines(self.graphicsImg, exterior, True, (0, 255, 255))

            for i in range(len(self.plantCenters2D[0])):
                # draw point on countur centers
                x = int(self.plantCenters2D[0, i])
                y = int(self.plantCenters2D[1, i])
                self.graphicsImg = cv.circle(
                    self.graphicsImg, (x, y), 3, (255, 0, 255), 5)

    def handleKey(self, sleepTime=0):
        key = cv.waitKey(sleepTime)
        # Close program with keyboard 'q'
        if key == ord('q'):
            cv.destroyAllWindows()
            exit(1)

    def bufferforlines(self, line, lineID):
        self.bufferzone = np.delete(self.bufferzone, lineID*self.buffersize, axis = 0)
        self.bufferzone = np.insert(self.bufferzone, lineID*self.buffersize + self.buffersize - 1, line, axis = 0)

        temp = self.bufferzone[lineID*self.buffersize: lineID*self.buffersize + self.buffersize]
        for i in range(self.buffersize):
            if np.all(temp[0] == (0, 0)):
                temp = np.delete(temp, obj = 0, axis = 0)
        new_line = np.mean(temp, axis = 0)

        return new_line
        # print("bufferzone", bufferzone)

    
    def get_3d_points(self, pixels):
        # print("intrinsics:", self.depth_intrinsics)
        fx = self.depth_intrinsics[0]
        fy = self.depth_intrinsics[1]
        cx = self.depth_intrinsics[2]
        cy = self.depth_intrinsics[3]
        threshold_low = 0.01
        threshold_high = 5 
        z_metric = []
        metric_points = []
        # import pdb
        # pdb.set_trace()
        for pixel_loc in pixels:
            xmin, ymin, xmax, ymax = self.generate_roi(pixel_loc)
            depthROI = self.primaryDepthImg[ymin:ymax, xmin:xmax]
            inRange = (threshold_low <= depthROI) & (depthROI <= threshold_high)
            averageDepth = np.mean(depthROI[inRange])
            print("averagedepth:", averageDepth)
            # print("averagedepth:", averageDepth)
            if averageDepth >= threshold_low:                
                # Convert to meters
                # averageDepth /= 1000

                # pixels_new.append(pixel_loc)
                # z_metric.append(averageDepth)
                x_metric = averageDepth*((cx - pixel_loc[0])/fx)
                y_metric = averageDepth*((cy - pixel_loc[1])/fy)
                z_metric = averageDepth
                metric_points.append([x_metric, y_metric, z_metric])
                # metric_points.append([z_metric, x_metric, y_metric])

        # print("pixels_new", pixels_new)
        # print("lines_metric", metric_points)

        #Here I'm getting depth of a single pixel, doing it on a ROI is better
        # new_pixels = np.flip(pixels, axis=1).astype(int)
        # z_metric_all = self.primaryDepthImg[tuple(new_pixels.T)]
                
        return np.array(metric_points)
    
    def generate_roi(self, center):
        delta = 5 # 5 pixels around the center to
        center = center.astype(int)
        x_center_new = min(max(center[0], delta), self.primaryDepthImg.shape[1] - delta)
        y_center_new = min(max(center[1], delta), self.primaryDepthImg.shape[0] - delta)

        return (x_center_new-delta, y_center_new-delta, x_center_new+delta, y_center_new+delta) #xmin ymin xmax y max
    
    def remove_duplicates(self, points, threshold=20):
        diff_elements = np.diff(points[:,0])
        closest_points = np.where(abs(diff_elements)<threshold)
        if closest_points[0].size>0:
            for closest in closest_points[0]:
                new_point = np.mean(points[closest:closest+2,:], axis=0)
                points[closest,:] = new_point
                points[closest+1,:] = math.nan
            points = points[~np.isnan(points)].reshape(-1,2)

        print("points", points)
        points_new = deepcopy(points)

        # points_new = np.zeros_like(points)
        points_new[0,:] = points[0,:]
        points_new[1,:] = points[2,:]
        points_new[2,:] = points[1,:]
        # points_new[3,:] = points[3,:]
        # points_new[3:,:] = points[3:,:]
        print("points_new", points_new)
        return points_new
    def get3dpoints(self, depthImg, pixels):
        depths = depthImg[pixels[0], pixels[1]]
        print("depth:", depths)
        points_in_3d = np.array([self.imgWidth- pixels[1], pixels[0], depths]).T
        return points_in_3d        
    def get_pixels(self, t_i, b_i):
        pixels = np.array([[0, t_i], [self.imgHeight - 1, b_i]])
        pixels = pixels.astype('int32')
        return pixels

    