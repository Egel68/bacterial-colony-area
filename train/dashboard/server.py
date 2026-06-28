import asyncio
import json
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

HERE = Path(__file__).parent
app = FastAPI()


@app.on_event("startup")
async def _capture_loop():
    global _loop
    _loop = asyncio.get_running_loop()

_loop: asyncio.AbstractEventLoop | None = None

_metrics_store: dict[str, list] = {
    "epochs": [],
    "train": [],
    "val": [],
}
_clients: set[WebSocket] = set()


def update(epoch: int, total: int, train_metrics: dict, val_metrics: dict, elapsed: float):
    _metrics_store["epochs"].append(epoch)
    _metrics_store["train"].append({**train_metrics, "epoch": epoch})
    _metrics_store["val"].append({**val_metrics, "epoch": epoch})
    data = json.dumps({
        "epoch": epoch,
        "total": total,
        "train": train_metrics,
        "val": val_metrics,
        "elapsed": round(elapsed, 2),
    })
    for ws in set(_clients):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(data), _loop)
        except Exception:
            _clients.discard(ws)


@app.get("/", response_class=HTMLResponse)
async def index():
    template = HERE / "templates" / "index.html"
    return HTMLResponse(template.read_text())


@app.get("/history")
async def history():
    return _metrics_store


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _clients.discard(websocket)


def run_server(port: int = 8765):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
