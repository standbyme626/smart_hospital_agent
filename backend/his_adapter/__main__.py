from __future__ import annotations

import uvicorn

from .config import AdapterConfig


def main() -> None:
    config = AdapterConfig.from_env()
    uvicorn.run("his_adapter.mcp_server:app", host=config.host, port=config.port)


if __name__ == "__main__":
    main()
