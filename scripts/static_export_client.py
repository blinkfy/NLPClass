"""Invoke synchronous read-only FastAPI routes without sockets or an event loop."""

import inspect
import json
import sys
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from starlette.requests import Request

from nlplab.core.exceptions import DataValidationError, NoValidPathError, ResourceUnavailableError
from nlplab.data.course import file_sha256


class ExportResponse:
    def __init__(self, response):
        self.status_code = response.status_code
        self.headers = response.headers
        self.content = (
            Path(response.path).read_bytes()
            if isinstance(response, FileResponse)
            else response.body
        )

    @property
    def text(self):
        return self.content.decode("utf-8")

    def json(self):
        return json.loads(self.content)


class StaticExportClient:
    def __init__(self, app):
        self.app = app
        self._hash_readers = []
        self._digests = {}

        def cached_digest(path):
            path = Path(path).resolve()
            stat = path.stat()
            key = (path, stat.st_size, stat.st_mtime_ns)
            if key not in self._digests:
                self._digests[key] = file_sha256(path)
            return self._digests[key]

        self._cached_digest = cached_digest
        self._entered = False

    def _cache_hash_readers(self):
        # A frozen file needs one hash check per export, rather than one per
        # document panel. Changed size/mtime invalidates the export-local cache.
        for name, module in list(sys.modules.items()):
            if (
                name.startswith("nlplab.")
                and module is not None
                and getattr(module, "file_sha256", None) is file_sha256
            ):
                self._hash_readers.append(module)
                module.file_sha256 = self._cached_digest

    def __enter__(self):
        self._entered = True
        self._cache_hash_readers()
        return self

    def __exit__(self, *args):
        for module in self._hash_readers:
            module.file_sha256 = file_sha256
        self._hash_readers.clear()
        self._entered = False

    def get(self, path, params=None):
        if self._entered:
            self._cache_hash_readers()
        path = urlsplit(path).path
        params = params or {}
        query = urlencode(params, doseq=True).encode()
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": query,
            "root_path": "",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 0),
            "headers": [],
            "app": self.app,
            "router": self.app.router,
        }
        try:
            for route in self.app.routes:
                if "GET" not in getattr(route, "methods", []):
                    continue
                match = route.path_regex.fullmatch(path)
                if not match:
                    continue
                scope["path_params"] = match.groupdict()
                values = {}
                for name, param in inspect.signature(
                    route.endpoint, eval_str=True
                ).parameters.items():
                    if name == "request":
                        values[name] = Request(scope)
                        continue
                    raw = match.groupdict().get(name, params.get(name, inspect.Parameter.empty))
                    if raw is inspect.Parameter.empty:
                        raw = (
                            param.default.default
                            if isinstance(param.default, FieldInfo)
                            else param.default
                        )
                    if raw is inspect.Parameter.empty:
                        raise ValueError(f"Missing export route parameter: {name}")
                    annotation = param.annotation
                    values[name] = (
                        raw
                        if annotation is inspect.Parameter.empty
                        else TypeAdapter(annotation).validate_python(raw)
                    )
                response = route.endpoint(**values)
                if inspect.isawaitable(response):
                    response.close()
                    raise ValueError("Async routes are excluded from static export")
                return ExportResponse(
                    response if isinstance(response, Response) else JSONResponse(response)
                )
            return ExportResponse(JSONResponse({"detail": "Not Found"}, status_code=404))
        except ResourceUnavailableError as error:
            return ExportResponse(
                JSONResponse(
                    {"error": {"code": "resource_unavailable", "message": str(error)}},
                    status_code=503,
                )
            )
        except (DataValidationError, NoValidPathError, ValidationError, ValueError) as error:
            return ExportResponse(
                JSONResponse(
                    {"error": {"code": "invalid_data", "message": str(error)}}, status_code=422
                )
            )
        except HTTPException as error:
            return ExportResponse(
                JSONResponse({"detail": error.detail}, status_code=error.status_code)
            )
