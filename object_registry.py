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
    

    def register(self,t:Type[T],name:str , item: T,overwrite :bool = False,immutable:bool = False) -> None:
        """
        
        Register an item with a given name.

        :param t: type prefix for the item
        :param name: identifier for the item - must be unique within the prefix
        :param item: the item to register
        :param overwrite: whether to overwrite an existing item with the same name
        :param immutable: whether the item should be treated as immutable (not modifiable)
        
        """
        uri = self._get_urit(t,name)
        was_overwrite = overwrite and uri in self._items

        if not overwrite and uri in self._items:
            raise ValueError(f"Item '{uri}' already registered,use a differnet name")
        
        self._items[uri]= Item(item, immutable = immutable)

        action = "overwrote" if was_overwrite else "registered"
        logger.info (f"Regsitry:{action}{uri}{immutable},total:{len(self._item)} items)")

        def register_multiple()