import os
from pathlib import Path


def is_valid_directoy(path: os.PathLike[str], must_exist: bool = False):
    path = Path(path)
    if path.exists() and path.is_dir():
        return True
    if must_exist:
        return False
    else:
        return path.suffix == ""
