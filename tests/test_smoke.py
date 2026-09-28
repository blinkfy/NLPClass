"""Phase 1: 工程安装和 Web 骨架可用。"""

from fastapi.testclient import TestClient

from app.main import create_app
from nlplab import __version__


def test_health() -> None:
    response = TestClient(create_app()).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
