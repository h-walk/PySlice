from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
import pickle
import hashlib
from .wf_data import WFData

logger = logging.getLogger(__name__)

@dataclass
class HAADFData(WFData):
    # inherit all attributes from parent object
    def __init__(self, WFData) -> object:
        self.__class__ = type(WFData.__class__.__name__,
                              (self.__class__, WFData.__class__),
                              {})
        self.__dict__ = WFData.__dict__

    def ADF(self, collection_angle: float = 45, preview: bool = False) -> np.ndarray:
        xs=np.asarray(sorted(list(set(self.probe_positions[:,0]))))
        ys=np.asarray(sorted(list(set(self.probe_positions[:,1]))))
        adf=np.zeros((len(xs),len(ys)))
        q=np.sqrt(self.kxs[:,None]**2+self.kys[None,:]**2)
        #print(np.shape(self.wavefunction_data),np.shape(q))
        radius = (collection_angle * 1e-3) / self.probe.wavelength * 2 * np.pi
        mask=np.zeros(q.shape) ; mask[q>radius]=1
        for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                dxy=np.sqrt( np.sum( (self.probe_positions-np.asarray([x,y])[None,:])**2,axis=1 ) )
                p=np.argmin(dxy)
                exits=self.wavefunction_data[p,:,:,:,-1] # which probe position, all frames, kx, ky, last layer
                if preview and i==0 and j==0:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.imshow(np.mean(np.absolute(exits),axis=0)**.1*(1-mask), cmap="inferno")
                    plt.show()
                #print(np.shape(exits),p,np.sum(np.absolute(exits)))
                collected = np.mean(np.sum( np.absolute(exits*mask[None,:,:]),axis=(1,2)))
                adf[i,j]=collected #; print(collected)
        return adf