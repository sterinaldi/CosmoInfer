from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh
cimport cython

cdef class CosmologicalParameters:

    def __cinit__(self,double h, double om, double ol, double w0, double w1):
        self.h = h
        self.om = om
        self.ol = ol
        self.w0 = w0
        self.w1 = w1
        self.__LALCosmologicalParameters = XLALCreateCosmologicalParameters(self.h,self.om,self.ol,self.w0,self.w1,0.0)
    
    cpdef void SetH(self, double h):
        self.h = h
        self.__LALCosmologicalParameters.h = h
    
    cpdef void SetOM(self, double om):
        self.om = om
        self.__LALCosmologicalParameters.om = om

    cpdef void SetOL(self, double ol):
        self.ol = ol
        self.__LALCosmologicalParameters.ol = ol

    cpdef double HubbleParameter(self,double z):
        return XLALHubbleParameter(z, self.__LALCosmologicalParameters)

    cpdef double LuminosityDistance(self, double z):
        return XLALLuminosityDistance(self.__LALCosmologicalParameters,z)

    cpdef double HubbleDistance(self):
        return XLALHubbleDistance(self.__LALCosmologicalParameters)

    cpdef double IntegrateComovingVolumeDensity(self, double zmax):
        return XLALIntegrateComovingVolumeDensity(self.__LALCosmologicalParameters,zmax)

    cpdef double IntegrateComovingVolume(self, double zmax):
        return XLALIntegrateComovingVolume(self.__LALCosmologicalParameters,zmax)

    cpdef double UniformComovingVolumeDensity(self, double z):
        return XLALUniformComovingVolumeDensity(z, self.__LALCosmologicalParameters)

    cpdef double UniformComovingVolumeDistribution(self, double z, double zmax):
        return XLALUniformComovingVolumeDistribution(self.__LALCosmologicalParameters, z, zmax)

    cpdef double ComovingVolumeElement(self,double z):
        return XLALComovingVolumeElement(z, self.__LALCosmologicalParameters)

    cpdef double ComovingVolume(self,double z):
        return XLALComovingVolume(self.__LALCosmologicalParameters, z)

    cpdef void DestroyCosmologicalParameters(self):
        XLALDestroyCosmologicalParameters(self.__LALCosmologicalParameters)
        return
