import urllib.request

from importlib import resources
from pathlib import Path
from typing import Union

_URL_REPO = "https://raw.githubusercontent.com/microsoft/pyright/main/"
_URL_MESSAGES = (
    f"{_URL_REPO}packages/pyright-internal/src/localization/package.nls.en-us.json"
)


def _download(url: str, file: Union[str, Path]) -> None:
    response = urllib.request.urlopen(url)
    with open(file, "wb") as f:
        f.write(response.read())


def _update_data_files() -> None:
    """Download data files from GitHub. These provide the schema for the parsing
    of the Pyright CLI output. Ideally, we'd interop with the Pyright code directly
    or consume the language server protocol, but this works well for now.
    """
    base_dir = resources.files("nicepyright") / "data"
    file = Path(str(base_dir.joinpath("messages.json")))
    _download(_URL_MESSAGES, file)


if __name__ == "__main__":
    _update_data_files()
