from pathlib import Path


class DataPaths:
    _registry: dict[str, Path] = {}
    _scanned = False

    @classmethod
    def autopath(cls, root_path="/kaggle/input", suffixes=None, refresh=False):
        """
        Scans the environment for data files while ignoring source code.
        """
        if cls._scanned and not refresh:
            return cls

        cls._registry.clear()
        # Define what we ARE looking for
        valid_suffixes = set(suffixes or [".obo", ".tsv", ".csv", ".txt", ".parquet"])
        # Define what we ARE NOT looking for
        blacklist_ext = {".py", ".pyc", ".ipynb", ".sh", ".md"}

        # Search /kaggle/input and the current working directory
        search_dirs = [Path(root_path), Path.cwd()]

        for base in search_dirs:
            if not base.exists():
                continue

            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() not in blacklist_ext:
                    if not suffixes or path.suffix.lower() in valid_suffixes:
                        # Register by full filename and by stem (filename without .ext)
                        cls._registry[path.name] = path.resolve()
                        cls._registry[path.stem] = path.resolve()

        cls._scanned = True
        return cls

    @classmethod
    def get(cls, name: str) -> str:
        """
        Returns the absolute string path for a file.
        """
        # Priority 1: Exact match (best for 'go-basic.obo')
        if name in cls._registry:
            return str(cls._registry[name])

        # Priority 2: Substring match excluding Python files
        matches = [p for k, p in cls._registry.items() if name in k]

        if not matches:
            raise FileNotFoundError(f"No file found matching '{name}' in registry.")

        if len(matches) > 1:
            # Heuristic: pick the one with the shortest name (most likely the direct hit)
            matches.sort(key=lambda p: len(p.name))

        return str(matches[0])

    def exists(cls, name: str) -> bool:
        return name in cls._registry
