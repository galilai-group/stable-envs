import importlib
import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def isolated_call(module: str, function: str, args: tuple) -> Any:
    """Run source capture without sharing its GL context lifecycle with the candidate."""
    with TemporaryDirectory(prefix="source-pixels-") as directory:
        request, response = Path(directory) / "request.pkl", Path(directory) / "response.pkl"
        request.write_bytes(pickle.dumps((module, function, args)))
        subprocess.run(
            [sys.executable, "-m", __name__, str(request), str(response)],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            timeout=300,
        )
        return pickle.loads(response.read_bytes())


if __name__ == "__main__":
    # Requests are generated locally by isolated_call, never read from fixture data.
    module, function, args = pickle.loads(Path(sys.argv[1]).read_bytes())
    result = getattr(importlib.import_module(module), function)(*args)
    Path(sys.argv[2]).write_bytes(pickle.dumps(result))
