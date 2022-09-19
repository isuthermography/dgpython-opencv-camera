import sys

from dataguzzler_python.dgpy import Module as dgpy_Module

import spatialnde2 as snde

import cv2
import numpy as np

from typing import ClassVar

from matplotlib import pyplot as plt


class OpenCVCalibration(object, metaclass=dgpy_Module):
    recdb: ClassVar[snde.recdatabase] = None
    chanptr = None
    rec: ClassVar[snde.multi_ndarray_recording] = None
    camchannel: str = None
    objpoints: list = None
    imgpoints: list = None
    checkershape = None
    brightfield = None
    darkfield = None
    mtx = None
    dist = None
    newmtx = None
    roi = None

    def __init__(self, module_name: str, recdb: snde.recdatabase, camchannel: str, nchkw: int, nchkh: int):
        self.module_name = module_name
        self.recdb = recdb
        self.camchannel = camchannel
        self.objpoints = []
        self.imgpoints = []
        self.checkershape = None
        self.nchkw = nchkw
        self.nchkh = nchkh
        self.rec = None
        self.chanptr = None
        self.brightfield = None
        self.darkfield = None
        self.mtx = None
        self.dist = None
        self.newmtx = None        
        self.roi = None


        self.brightfield = np.zeros((3,3), dtype=np.uint8) + 255
        self.darkfield = np.zeros((3,3), dtype=np.uint8)
        self.mtx = np.array([[1,0,3/2],[0,1,3/2],[0,0,1]], dtype=np.float32)
        self.newmtx = self.mtx.copy()
        self.dist = np.array([0,0,0,0,0], dtype=np.float32)
        self.roi = np.array([0,0,-1,-1], dtype=np.int32)


        transact = self.recdb.start_transaction()
        self.chanptr = self.recdb.define_channel(self.module_name, "main", self.recdb.raw())
        self.recdb.end_transaction(transact)

        self.SetCalibration()

    def SetCalibration(self):
        if self.brightfield is not None and self.darkfield is not None and self.mtx is not None and self.dist is not None and self.newmtx is not None and self.roi is not None:                

            if self.brightfield.dtype == np.uint8:
                dtype = snde.SNDE_RTN_UINT8
            elif self.brightfield.dtype == np.uint16:
                dtype = snde.SNDE_RTN_UINT16
            elif self.brightfield.dtype == np.float16:
                dtype = snde.SNDE_RTN_FLOAT16
            elif self.brightfield.dtype == np.float32:
                dtype = snde.SNDE_RTN_FLOAT32
            elif self.brightfield.dtype == np.float64:
                dtype = snde.SNDE_RTN_FLOAT64
            else:
                raise Exception('Unknown Data Type')

            transact = self.recdb.start_transaction()
            self.rec = snde.create_multi_ndarray_recording(self.recdb, self.chanptr, self.recdb.raw(), 6)
            self.rec.define_array(0, dtype, "brightfield")
            self.rec.define_array(1, dtype, "darkfield")
            self.rec.define_array(2, snde.SNDE_RTN_FLOAT32, "cam_mtx")
            self.rec.define_array(3, snde.SNDE_RTN_FLOAT32, "cam_dist")
            self.rec.define_array(4, snde.SNDE_RTN_FLOAT32, "cam_newmtx")
            self.rec.define_array(5, snde.SNDE_RTN_INT32, "cam_roi")
            globalrev = self.recdb.end_transaction(transact)
            self.rec.allocate_storage("brightfield", self.brightfield.shape)
            self.rec.allocate_storage("darkfield", self.darkfield.shape)
            self.rec.allocate_storage("cam_mtx", self.mtx.shape)
            self.rec.allocate_storage("cam_dist", self.dist.shape)
            self.rec.allocate_storage("cam_newmtx", self.newmtx.shape)
            self.rec.allocate_storage("cam_roi", self.roi.shape)
            self.rec.metadata = snde.immutable_metadata()
            self.rec.mark_metadata_done()
            self.rec.reference_ndarray('brightfield').data()[:] = self.brightfield
            self.rec.reference_ndarray('darkfield').data()[:] = self.darkfield
            self.rec.reference_ndarray('cam_mtx').data()[:] = self.mtx
            self.rec.reference_ndarray('cam_dist').data()[:] = self.dist
            self.rec.reference_ndarray('cam_newmtx').data()[:] = self.newmtx
            self.rec.reference_ndarray('cam_roi').data()[:] = self.roi
            self.rec.mark_data_ready()
            globalrev.wait_complete()
            return True

        else:
            print("Brightfield, Darkfield, and Camera Calibration is Required")
            return False


    def CaptureBrightfieldImage(self):
        mon: snde.monitor_globalrevs = self.recdb.start_monitoring_globalrevs()
        try:            
            rev: snde.globalrevision = mon.wait_next(self.recdb)
        finally:
            mon.close(self.recdb)

        rec = rev.get_recording_ref(self.camchannel)
        self.brightfield = rec.data()
        return True

    def CaptureDarkfieldImage(self):
        mon: snde.monitor_globalrevs = self.recdb.start_monitoring_globalrevs()
        try:            
            rev: snde.globalrevision = mon.wait_next(self.recdb)
        finally:
            mon.close(self.recdb)

        rec = rev.get_recording_ref(self.camchannel)
        self.darkfield = rec.data()
        return True

    def ProcessCameraCalibration(self):
        if len(self.objpoints) == 0 or len(self.imgpoints) == 0:
            print("Must capture checkerboard images first")
            return False

        mon: snde.monitor_globalrevs = self.recdb.start_monitoring_globalrevs()
        try:            
            rev: snde.globalrevision = mon.wait_next(self.recdb)
        finally:
            mon.close(self.recdb)

        rec = rev.get_recording_ref(self.camchannel)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.checkershape[::-1], None, None)

        if ret:
            newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (self.checkershape[1], self.checkershape[0]), 1, (self.checkershape[1], self.checkershape[0]))
            self.newmtx = newmtx
            self.mtx = mtx
            self.dist = dist
            self.roi = np.array(roi)
            return True
        else:
            print("Calibration Failed")
            return False

    def CaptureCheckerboardImage(self, nchkw = None, nchkh = None, imgprocess=False, plot=True):
        if nchkw is None and self.nchkw is not None:
            nchkw = self.nchkw
        else:
            raise Exception("Must specify checkerboard width")
        if nchkh is None and self.nchkh is not None:
            nchkh = self.nchkh
        else:
            raise Exception("Must specify checkerboard height")

        mon: snde.monitor_globalrevs = self.recdb.start_monitoring_globalrevs()
        try:            
            rev: snde.globalrevision = mon.wait_next(self.recdb)
        finally:
            mon.close(self.recdb)

        rec = rev.get_recording_ref(self.camchannel)
        img = rec.data()

        if img.dtype.fields != None and all(elem in img.dtype.fields for elem in ['r','g','b']):
            # Type is RGB, we'll convert to 8-bit gray
            img8 = (img['r'] * 0.299 + img['g'] * 0.587 + img['b'] * 0.114).astype('uint8')
        elif img.dtype == np.uint16:
            # Type is uint16 gray, we'll convert to 8-bit gray
            img8 = (img/256).astype('uint8')
        elif img.dtype == np.uint8 and len(img.shape) == 3:
            # Type is probably BGR from OpenCV, we'll convert to 8-bit gray
            img8 = (img[:,:,2] * 0.299 + img[:,:,1] * 0.587 + img[:,:,0] * 0.114)
        elif img.dtype == np.uint8 and len(img.shape) == 2:
            # Type is already correct, we'll leave alone
            img8 = img
        else:
            print("Unknown Data Type")
            return False

        if self.checkershape is not None:
            if self.checkershape != img8.shape:
                print("Image Shape Has Changed -- Clearing Old Images")
                self.objpoints = []
                self.imgpoints = []
                self.checkershape = img8.shape
        else:
            self.checkershape = img8.shape

        if imgprocess:
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        else:
            flags = None
        ret, corners = cv2.findChessboardCorners(img8, (nchkw, nchkh), flags)

        if ret == False:
            print("Failed to Find Chessboard Pattern -- Try Calling CaptureImage(imgprocess=True) instead")
            return False

        objp = np.zeros((nchkw * nchkh, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nchkw, 0:nchkh].T.reshape(-1,2)

        corners2 = cv2.cornerSubPix(img8, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        self.objpoints.append(objp)
        self.imgpoints.append(corners2)

        if plot:
            plt.ion()
            plt.figure()
            plt.imshow(img8.T, origin='lower', cmap='gray')
            plt.plot(corners2[:,0,1], corners2[:,0,0], 'rx-')
            plt.show()

        return True


            

        


