import asyncio
import json


class ConnectionManager:
    def __init__(self):
        self._connections: list = []

    async def connect(self, ws) -> None:
        await ws.accept()
        self._connections.append(ws)

    async def disconnect(self, ws) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, data: dict) -> None:
        msg = json.dumps(data)
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)
