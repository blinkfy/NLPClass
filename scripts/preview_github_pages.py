"""Serve the static directory locally with correct MIME types for ES modules."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="docs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    SimpleHTTPRequestHandler.extensions_map.update(
        {".mjs": "text/javascript", ".wasm": "application/wasm"}
    )
    directory = Path(args.directory).resolve()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory=str(directory))
    )
    print(f"Static preview: http://127.0.0.1:{args.port}/ ({directory})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
