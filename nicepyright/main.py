import asyncio

from rich.align import Align
from rich.panel import Panel
from rich.text import Text

from nicepyright import con
from nicepyright.definitions import PyrightOutput


def watch() -> None:
    async def main() -> None:
        con.print(
            Panel(
                Align(Text("Starting pyright...", justify="center"), vertical="middle"),
                expand=True,
            ),
        )

        async for output in PyrightOutput.pyright_watch():
            con.print(output)

    asyncio.run(main())


if __name__ == "__main__":
    # y: float = "boo"
    # x: int = 4.5
    z: int = 4.5
    watch()


__all__ = ("watch",)
