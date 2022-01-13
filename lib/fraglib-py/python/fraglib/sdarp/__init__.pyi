from . import instance
from typing import Tuple, List, Dict, Optional

class SdarpDataWrapper:
    ...

def extended_restricted_fragments(data: SdarpDataWrapper, dominate: Optional[str], *, cpus: int = 0) -> Tuple[List[Tuple[List[int], int, int, int]], Dict[str, int]]:
    """
    Loads data according to `index`, preprocesses if requested and generates Extended Restricted Fragments.
    Dominates fragments if requests.  Returns a list of extended fragments and a dict containing
    statistics on generation.  Each extended fragment is a 4-tuple containing, in order, the path, time of earliest
    finish, time of latest start and total travel time.
    """
    ...

def preprocess_data(data: SdarpDataWrapper):
    ...


