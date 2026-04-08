"""OpenEnv server entrypoint for KAVACH-X."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import uvicorn


def _load_root_fastapi_app() -> Any:
    """Load FastAPI app from the repository root app.py."""
    root_app_path = Path(__file__).resolve().parents[1] / "app.py"
    spec = importlib.util.spec_from_file_location("kavach_x_root_app", root_app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load root app from {root_app_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "app"):
        raise RuntimeError("Root app.py does not expose variable 'app'.")
    return module.app


app = _load_root_fastapi_app()


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the KAVACH-X API server."""
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
