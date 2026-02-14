# Rend le dossier src importable


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch n'est pas installe. Installe les dependances via `pip install -r requirements.txt`."
        ) from exc
    return torch
