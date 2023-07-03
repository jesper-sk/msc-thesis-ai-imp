from pathlib import Path


def is_valid_directory(path: Path | str, must_exist: bool = False):
    path = Path(path)
    if path.exists() and path.is_dir():
        return True
    if must_exist:
        return False
    else:
        return path.suffix == ""


def validate_and_create_dir(path: Path | str) -> Path:
    if not is_valid_directory(path):
        raise ValueError(
            f"The path f{path} does not point to a valid directory location"
        )

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    return path


def validate_existing_dir(path: Path | str) -> Path:
    if not is_valid_directory(path, must_exist=True):
        raise ValueError(
            f"The path f{path} does not point to a valid existing directory location"
        )

    return Path(path)
