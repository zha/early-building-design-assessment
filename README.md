# Early building design assessment
A simulation workflow built for my Master's thesis at UofT
* To install:
```python
pip install git+https://github.com/zha/early-building-design-assessment.git
```
## Dev Status
- [x] Geometry
- [x] EnergyPlus
- [x] Radiance
- [ ] Draft comfort
- [ ] Post processing comfort


## Usage example
```python

from design_simulation import ModelInit, observer
from design_simulation import EnergyModel
from design_simulation import RadianceModel

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


model = ModelInit(working_dir = r"z:/sdfsdfsds", wea_dir = r"e:/CAN_ON_Toronto.716240_CWEC.epw")
observer(model)
model.zone_name = '4' 
model.zone_width = 3.3
model.zone_depth = 8
model.zone_height = 2
model.orientation = 'south'
model.WWR = 3
model.U_factor = 2.6
model.SHGC = 0.6
model.update()

```

## Acknowledgment
* JF, MT, LO (no particular order)
* Ladybug Tools, CM, MM
* NSERC, ON
* RDH
