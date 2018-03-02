import cv2
import numpy as np
import re
import math
from numpy.linalg import inv, norm


class Dlt(object):
    """docstring for Dlt."""
    def __init__(self, orgs,corrs):
        self.orgs = orgs
        self.corrs = corrs

    def average(org, corrs):
        averageOrg = np.zeros(2)
        averageCorr = np.zeros(2)
        for point in org:
            averageOrg[0] += point[0]
            averageOrg[1] += point[1]
        for point in corres:
            averageCorr[0] += point[0]
            averageCorr[1] += point[1]
        averageOrg /= len(org)
        averageCorr /= len(corrs)

        return averageOrg, averageCorr


    def __scale(points, average):
        sumOfDist = 0.0
        for point in points:
            sumOfDist += norm(point - average)
        s = math.sqrt(2) * len(points) / (sumOfDist)
        return s


    def __matrixT(points, average):
        s = scale(points, average)
        tX = s * (-average[0])
        tY = s * (-average[1])
        T = np.float64([[s, 0, tX], [0, s, tY], [0, 0, 1]])
        return T


    def __normalize(points, T):
        normalizedPoints = np.zeros((len(points), 3))
        i = 0
        for point in points:
            normalizedPoints[i] = np.transpose(T @ point)
            i += 1
        return normalizedPoints


    def __matrixA(normalizedOrg, normalizedCorr):
        A = np.zeros((len(normalizedOrg) * 2, 9))
        i = 0
        for index in range(0, len(A), 2):
            A[index] = np.float64([0, 0, 0, -normalizedOrg[i][0], -normalizedOrg[i][1], -1, normalizedCorr[i]
                                   [1] * normalizedOrg[i][0], normalizedCorr[i][1] * normalizedOrg[i][1], normalizedCorr[i][1]])
            A[index + 1] = np.float64([normalizedOrg[i][0], normalizedOrg[i][1], 1, 0, 0, 0, -normalizedCorr[i]
                                       [0] * normalizedOrg[i][0], -normalizedCorr[i][0] * normalizedOrg[i][1], -normalizedCorr[i][0]])
            i += 1
        return A


    def computeH(vT, orgT, corrT):
        averageOrg, averageCorr = average(src, dst)
        orgT = matrixT(src, averageOrg)
        corrT = matrixT(dst, averageCorr)
        normalizedOrg = normalize(src, orgT)
        normalizedCorr = normalize(dst, corrT)
        A = matrixA(normalizedOrg, normalizedCorr)
        U, S, vT = cv2.SVDecomp(A)

        normalizedH = np.zeros((3, 3))
        i = 0
        for index in range(0, len(normalizedH)):
            normalizedH[index][0] = vT[-1][i]
            normalizedH[index][1] = vT[-1][i + 1]
            normalizedH[index][2] = vT[-1][i + 2]
            i += 3
        H = inv(corrT) @ normalizedH @ orgT
        return H
