import sys
from threading import Thread

from dataguzzler_python.dgpy import Module as dgpy_Module
from dataguzzler_python.dgpy import InitCompatibleThread

import spatialnde2 as snde

import cv2

class OpenCVCamera(object, metaclass=dgpy_Module):
    recdb = None
    chanptr = None
    cameranum = None
    thread = None
    _quit = False

    def __init__(self, module_name, recdb, cameranum=0):
        self.module_name = module_name
        self.recdb = recdb
        self.cameranum = cameranum

        transact = recdb.start_transaction()
        self.chanptr = recdb.define_channel("/CAMERA%d" % (self.cameranum), "main", self.recdb.raw())
        recdb.end_transaction(transact)

        self.StartAcqusition()

    def AcquisitionThread(self):
        InitCompatibleThread(self, "_thread")

        vid = cv2.VideoCapture(self.cameranum)

        try:
            while not self._quit:
                ret, frame = vid.read()
                transact = self.recdb.start_transaction()
                rec = snde.create_recording_ref(self.recdb, self.chanptr, self.recdb.raw(), snde.SNDE_RTN_UINT8)
                globalrev = self.recdb.end_transaction(transact)
                rec.rec.metadata = snde.immutable_metadata()
                rec.rec.mark_metadata_done()
                rec.allocate_storage(frame.shape[0:2], False)
                outdata = rec.data()
                outdata[:,:] = frame.mean(axis=2)
                rec.rec.mark_data_ready()
                pass
        finally:
            vid.release()

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
        


    pass