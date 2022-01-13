from typing import overload
from . import apvrp
from . import sdarp

import utils.data

@overload
def unwrap(data: apvrp.ApvrpDataWrapper) -> utils.data.APVRP_Data:
    ...

@overload
def unwrap(data: sdarp.SdarpDataWrapper) -> utils.data.SDARP_Data:
    ...
