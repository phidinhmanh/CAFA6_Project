from pathlib import Path


class DataPaths:
    _registry: dict[str, Path] = {}
    _scanned = False

    @classmethod
    def autopath(cls, root_path="/kaggle/input", suffixes=None, refresh=False):
        if cls._scanned and not refresh:
            return cls

        cls._registry.clear()
        # Only look for data-related suffixes by default
        target_suffixes = set(suffixes or [".obo", ".tsv", ".csv", ".txt", ".parquet"])

        # Focus on Kaggle input and the current working directory
        search_dirs = [Path(root_path), Path.cwd()]

        for base in search_dirs:
            if not base.exists():
                continue

            for path in base.rglob("*"):
                # CRITICAL: Only register files with data suffixes
                if path.is_file() and path.suffix.lower() in target_suffixes:
                    # Store by filename for exact matching
                    cls._registry[path.name] = path.resolve()
                    # Also store by stem (filename without extension) for convenience
                    cls._registry[path.stem] = path.resolve()

        cls._scanned = True
        return cls

    @classmethod
    def get(cls, name: str) -> Path:
        """
        Retrieves a path using strict matching first, then filtered substring matching.
        """
        # 1. Try Exact Match (e.g., "go-basic.obo")
        if name in cls._registry:
            return cls._registry[name]

        # 2. Try to find matches that aren't source code
        matches = [
            path
            for key, path in cls._registry.items()
            if name in key and path.suffix not in (".py", ".pyc", ".ipynb")
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            # Heuristic: Pick the one where the name is the most similar in length
            # (Prevents "obo" matching "some_long_filename_with_obo_in_it.tsv")
            matches.sort(key=lambda p: len(p.name))
            return matches[0]

        raise FileNotFoundError(
            f"Could not find data file matching '{name}'. "
            f"Available files: {list(cls._registry.keys())[:10]}..."
        )
