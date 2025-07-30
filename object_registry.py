import logging
import dataclasses
import copy
from typing import Dict,List,Type,TypeVar,Any

T = TypeVar("T")
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Item:
    item:T
    immutable:bool =False

class ObjectRegistry:
    """

    Registry for storing and retrieving objects by name.

    This class implements the Singleton pattern so that registry instances are shared
    across the application. It provides methods for registering, retrieving, and
    managing objects in a type-safe manner.
    
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectRegistry, cls).__new__(cls)
            cls._instance._items = {}
        return cls._instance

    @staticmethod
    def _get_uri(t:Type[T],name:str)->str:
        return f"{str(t)}://{name}"
    
