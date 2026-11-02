"""Avvio del server: python main.py"""

import logging
import os

import uvicorn


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Log di debug degli agenti (verbalization.make_narrator): TRACE_LEVEL=WARNING per spegnerli
    logging.getLogger("trace").setLevel(os.getenv("TRACE_LEVEL", "INFO").upper())


# Fuori dal guard: con --reload uvicorn reimporta questo modulo nel processo figlio,
# e così il logging viene configurato anche lì.
setup_logging()


if __name__ == "__main__":
    uvicorn.run(
        "api.routes:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
        workers=1,          # InMemorySaver vive nel processo: con più worker si perdono le conversazioni
        log_config=None,    # usa il logging configurato sopra invece di quello di uvicorn
    )
