from pathlib import Path


class DataPaths:
    _registry: dict[str, Path] = {}
    _scanned = False

    @classmethod
    def autopath(
        cls,
        root_path="/kaggle/input",
        suffixes=None,
        refresh=False,
        max_files=200_000,
    ):
        """
        Auto-discover files under root_path and /kaggle/input only.
        """
        if cls._scanned and not refresh:
            return cls

        cls._registry.clear()
        suffixes = set(suffixes or [])

        roots = []
        root = Path(__file__).parent
        kaggle_root = Path("/kaggle/input")

        if root.exists():
            roots.append(root)

        if kaggle_root.exists() and kaggle_root not in roots:
            roots.append(kaggle_root)

        for base in roots:
            count = 0
            for path in base.rglob("*"):
                if not path.is_file():
                    continue

                if suffixes and path.suffix not in suffixes:
                    continue

                cls._registry[path.name] = path.resolve()

                count += 1
                if count >= max_files:
                    break

        cls._scanned = True
        return cls

    @classmethod
    def get(cls, name: str) -> Path:
        """
        Resolve file with smart disambiguation:
        - ignore tmp/mock/debug files
        - prefer shortest & cleanest match
        """
        if name in cls._registry:
            return cls._registry[name]

        def is_noise(p: Path) -> bool:
            bad_tokens = ("tmp", "mock", "test", "debug", "backup")
            n = p.name.lower()
            return n.startswith(".") or any(t in n for t in bad_tokens)

        # raw substring matches
        matches = [p for k, p in cls._registry.items() if name in k]

        # filter noise
        clean = [p for p in matches if not is_noise(p)]

        if len(clean) == 1:
            return clean[0]

        if len(clean) > 1:
            # heuristic: shortest name wins
            clean.sort(key=lambda p: (len(p.name), len(p.parts)))
            return clean[0]

        if matches:
            # fallback: all are noisy → pick best one
            matches.sort(key=lambda p: len(p.name))
            return matches[0]

        raise FileNotFoundError(f"File '{name}' not found")

        """
        Resolve a file by:
        1. exact filename
        2. unique substring match
        """
        if name in cls._registry:
            return cls._registry[name]

        matches = [p for k, p in cls._registry.items() if name in k]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous file '{name}'. Matches: {[p.name for p in matches[:5]]}"
            )

        raise FileNotFoundError(f"File '{name}' not found")

    @classmethod
    def exists(cls, name: str) -> bool:
        """
        Check if a file can be resolved without raising.
        """
        if name in cls._registry:
            return True

        matches = [k for k in cls._registry if name in k]
        return len(matches) == 1
