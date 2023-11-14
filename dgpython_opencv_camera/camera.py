import sys
from threading import Thread
import queue

from dataguzzler_python.dgpy import Module as dgpy_Module
from dataguzzler_python.dgpy import InitCompatibleThread

import spatialnde2 as snde

import cv2
import numpy as np
import ctypes

# OpenCV DirectShow Driver Doesn't Report The Correct Value For Auto Params When In Auto Mode
# Need to report that these were set to auto until we have changed them, then it is kicked out
# of auto mode automatically.  There is no way to detect if they are in auto mode.
AutoParams=['Exposure', 'WhiteBalance']

ParamDict={
    "Width": (cv2.CAP_PROP_FRAME_WIDTH, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-width", val), "Width of the Frame"),
    "Height": (cv2.CAP_PROP_FRAME_HEIGHT, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-height", val), "Height of the Frame"),
    "FPS": (cv2.CAP_PROP_FPS, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("FPS", val), "camera_params-frame_rate"),
    "Brightness": (cv2.CAP_PROP_BRIGHTNESS, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-brightness", val), "Brightness of the Image"),
    "Contrast": (cv2.CAP_PROP_CONTRAST, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-contrast", val), "Contrast of the Image"),
    "Hue": (cv2.CAP_PROP_HUE, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-hue", val), "Hue of the Image"),
    "Saturation": (cv2.CAP_PROP_SATURATION, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-saturation", val), "Saturation of the Image"),
    "Gain": (cv2.CAP_PROP_GAIN, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-gain", val), "Gain of the Image"),
    "Exposure": (cv2.CAP_PROP_EXPOSURE, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-exposure", val), "Exposure"),
    "Sharpness": (cv2.CAP_PROP_SHARPNESS, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-sharpness", val), "Sharpness of the Image"),
    "Gamma": (cv2.CAP_PROP_GAMMA, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-gamma", val), "Gammma of the Image"),
    "BacklightComp": (cv2.CAP_PROP_BACKLIGHT, bool, lambda val: bool(val), lambda val: int(val), lambda val: snde.metadatum_bool("camera_params-backlight_comp", val), "Backlight Compensation"),
    "WhiteBalance": (cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, int, lambda val: val, lambda val: val, lambda val: snde.metadatum_int("camera_params-white_balance", val), "Color Temperature"),
}


class ParamAttribute(object):
    # Descriptor class. Instances of this are returned
    # by __getattribute__ on the CompuScope class
    # for each of the parameters (including with
    # numbered suffixes) in ParamDict
    # https://docs.python.org/2/howto/descriptor.html
    Name=None
    
    def __init__(self,Name,doc):
        self.Name=Name
        self.__doc__=doc
        pass
    
    def __get__(self,obj,type=None):
        return obj.GetParam(self.Name)

    def __set__(self,obj,value):
        obj.SetParam(self.Name,value)
        pass

    pass
    

def add_parameters(cls):
    for paramname in ParamDict:            
        # Create Descriptor for this parameter -- this is so help() works.
        
        descr = ParamAttribute(paramname,ParamDict[paramname][5])
        # Need to manually wrap the descriptor because dgpy.Module
        # can't otherwise control descriptor access -- 
        # because the descriptor is called within object.__getattribute__()
        setattr(cls,paramname,descr)        
        pass

    
    return cls


@add_parameters
class OpenCVCamera(object, metaclass=dgpy_Module):
    recdb = None
    chanptr = None
    cameranum = None
    capdevice = None
    thread = None
    _quit = False
    _paramvals = None
    _queue = None
    _settings = False

    def __init__(self, module_name, recdb, cameranum=0, capdevice = cv2.CAP_ANY, enablesettings=False):
        self.module_name = module_name
        self.recdb = recdb
        self.cameranum = cameranum
        self.capdevice = capdevice
        self._queue = queue.Queue()
        self._settings = False
        self.enablesettings = enablesettings

        transact = recdb.start_transaction()
        self.chanptr = recdb.define_channel(module_name, "main", self.recdb.raw())
        recdb.end_transaction(transact)

        # We need to add an initial frame to get the data type or else the averaging math channel will fail because it will see the null pointer
        transact = self.recdb.start_transaction()
        rec = snde.create_ndarray_ref(self.recdb, self.chanptr, self.recdb.raw(), snde.SNDE_RTN_UINT8)
        globalrev = self.recdb.end_transaction(transact)
        rec.rec.metadata = snde.immutable_metadata()
        rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-axis0_coord', 'Y Position'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-axis1_coord', 'X Position'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-axis2_coord', 'Channel'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-ampl_coord', 'Intensity'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-ampl_units', 'Arb'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis0_offset', -1, 'pixels'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis1_offset', -1, 'pixels'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis2_offset', 0, 'unitless'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis0_scale', 1, 'pixels'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis1_scale', 1, 'pixels'))
        rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis2_scale', 1, 'unitless'))
        rec.rec.mark_metadata_done()
        rec.allocate_storage([2, 2, 3], True)
        outdata = rec.data()
        outdata[:] = np.zeros([2,2,3], dtype=np.uint8)
        rec.rec.mark_data_ready()

        self.StartAcquisition()

    def ShowCameraSettings(self):
        self._settings = True

    def GetParam(self, name):
        return ParamDict[name][2](self._paramvals[name])

    def SetParam(self, name, value):
        self._queue.put((name, ParamDict[name][0], ParamDict[name][3](value)))
        self._queue.join()
        return self.GetParam(name)

    def AcquisitionThread(self):
        InitCompatibleThread(self, "_thread")

        vid = cv2.VideoCapture(self.cameranum, self.capdevice)
        
        self._paramvals = {}
        for key in ParamDict:
            self._paramvals[key] = vid.get(ParamDict[key][0])            

        try:
            while not self._quit:
                if self.enablesettings and self._settings:
                    vid.set(cv2.CAP_PROP_SETTINGS, 1)
                    self._settings = False
                while not self._queue.empty():
                    item = self._queue.get(False)
                    vid.set(item[1], item[2])
                    self._paramvals[item[0]] = vid.get(item[1])
                    self._queue.task_done()
                ret, frame = vid.read()
                if ret:
                    transact = self.recdb.start_transaction()
                    rec = snde.create_ndarray_ref(self.recdb, self.chanptr, self.recdb.raw(), snde.SNDE_RTN_UINT8)
                    globalrev = self.recdb.end_transaction(transact)
                    rec.rec.metadata = snde.immutable_metadata()
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-axis0_coord', 'Y Position'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-axis1_coord', 'X Position'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-axis2_coord', 'Channel'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-ampl_coord', 'Intensity'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_str('ande_array-ampl_units', 'Arb'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis0_offset', frame.shape[0]/(-2), 'pixels'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis1_offset', frame.shape[1]/(-2), 'pixels'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis2_offset', 0, 'unitless'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis0_scale', 1, 'pixels'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis1_scale', 1, 'pixels'))
                    rec.rec.metadata.AddMetaDatum(snde.metadatum_dblunits('ande_array-axis2_scale', 1, 'unitless'))
                    if self.enablesettings:
                        for key in ParamDict:
                            self._paramvals[key] = vid.get(ParamDict[key][0]) 
                    for key in ParamDict:
                        rec.rec.metadata.AddMetaDatum(ParamDict[key][4](ParamDict[key][1](ParamDict[key][2](self._paramvals[key]))))
                    rec.rec.mark_metadata_done()
                    rec.allocate_storage(frame.shape, True)
                    outdata = rec.data()
                    outdata[:] = frame
                    rec.rec.mark_data_ready()
                    globalrev.wait_complete()
        finally:
            self.vid.release()

    def StartAcquisition(self):
        if self.thread is not None:
            if self.thread.isAlive():
                sys.stderr.write("Warning: Acquisition Thread Already Running\n")
                sys.stderr.flush()
                return
        
        self._quit = False
        self.thread = Thread(target=self.AcquisitionThread, daemon=True)
        self.thread.start()

    def RestartAcquisition(self):
        if self.thread is not None:
            self._quit = True
            while self.thread.isAlive():
                pass    

        self.StartAcquisition()

    def StopAcquisition(self):
        if self.thread is not None:
            self._quit = True
            while self.thread.isAlive():
                pass           


    pass