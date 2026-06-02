import sys

# On Windows the default ProactorEventLoop surfaces ConnectionResetError when
# a client closes an HTTP connection without a graceful TCP shutdown (RST vs FIN).
# SelectorEventLoop handles this silently and has no downside for a local
# single-user app.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(data)
    except WebSocketDisconnect:
        pass
