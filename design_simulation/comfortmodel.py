import numpy as np
class ComfortModel:
    def __init__(self, initmodel, energymodel, radiancemodel):
        self._initmodel = initmodel
        self._energymodel = energymodel
        self._radiancemodel = radiancemodel

    @property
    def initmodel(self):   # grab the initial model from
        return self._initmodel

    @property
    def energymodel(self):
        return self._energymodel

    @property
    def radiancemodel(self):
        return self._radiancemodel

    @property
    def LW_MRT(self):
        try: return self._LW_MRT
        except:
            vf_array = np.array(list(self.initmodel.viewfactor.values()))
            vf_array = vf_array.reshape(vf_array.shape[0], -1).T
            surfacetemps = np.array(list(self.energymodel.result.surfacetemps.values()))
            self._LW_MRT = np.matmul(vf_array, (surfacetemps + 273.15) ** 4) ** 0.25 - 273.15
            return self._LW_MRT
