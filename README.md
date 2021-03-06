[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)
# A simulation tool for early building design assessment
A simulation workflow built for my Master's thesis at UofT
##  Installation:
```
pip install git+https://github.com/zha/early-building-design-assessment.git
```
May also need to install forked honeybee/ladybug repos:
```
pip install git+https://github.com/zha/honeybee-energy.git
pip install git+https://github.com/zha/honeybee.git
```

## Dev Status
- [x] Geometry
- [x] EnergyPlus
- [x] Radiance
- [ ] Draft comfort
- [ ] PMV calculation
- [ ] Comfort assessment
- [ ] Clean up
- [ ] Better documentation

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
* Ladybug Tools :beetle::honeybee:, CM, MR
* NSERC, ON
* DH from RDH
