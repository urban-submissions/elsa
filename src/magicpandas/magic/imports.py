from typing import TypeVar
T = TypeVar('T')
class Imports:
    def __getitem__(self, item: T) -> T:
        ...


imports = Imports()
