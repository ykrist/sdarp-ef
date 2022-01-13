from . import SdarpDataWrapper

def load(index : int) -> SdarpDataWrapper:
    """
    Loads data according to `index` and preprocesses if requested.
    """
    ...

def index_to_name(index : int) -> str:
    """
    Get the instance name by index
    """
    ...

def name_to_index(index : str) -> int:
    """
    Convert the instance name to index
    """
    ...

def len() -> int:
    """
    Get the number of instances in this dataset
    """
    ...