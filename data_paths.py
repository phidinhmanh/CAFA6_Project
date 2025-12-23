from pathlib import Path


class DataPaths:
    _registry: dict[str, Path] = {}
    _scanned = False

    @classmethod
    def autopath(cls, root_path="/kaggle/input", suffixes=None, refresh=False):
        if cls._scanned and not refresh:
            return cls

        cls._registry.clear()
        # Only allow these specific data extensions
        allowed_suffixes = {".obo", ".tsv", ".csv", ".parquet", ".txt"}

        # Search /kaggle/input and your current working directory
        search_dirs = [Path(root_path), Path.cwd()]

        for base in search_dirs:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                # BLOCK: Ignore any python files or hidden folders
                if path.suffix.lower() == ".py" or path.name.startswith("."):
                    continue

                if path.is_file() and path.suffix.lower() in allowed_suffixes:
                    # Register by full filename (go-basic.obo) and stem (go-basic)
                    cls._registry[path.name] = path.resolve()
                    cls._registry[path.stem] = path.resolve()

        cls._scanned = True
        return cls

    @classmethod
    def get(cls, name: str) -> str:
        # Priority 1: Exact match (matches 'go-basic.obo' or 'go-basic')
        if name in cls._registry:
            return str(cls._registry[name])

        # Priority 2: Safe substring match (filtering out any accidentally caught .py)
        matches = [p for k, p in cls._registry.items() if name in k]

        if not matches:
            raise FileNotFoundError(f"Could not find a data file matching '{name}'.")

        # Pick the shortest match (usually the most specific one)
        matches.sort(key=lambda p: len(p.name))
        return str(matches[0])

    @classmethod
    def exists(cls, name: str) -> bool:
        return name in cls._registry
