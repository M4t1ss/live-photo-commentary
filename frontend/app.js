const { invoke } = window.__TAURI__.core;

const backendStatusEl = document.getElementById("backend-status");
const wsStatusEl      = document.getElementById("ws-status");
const echoEl          = document.getElementById("echo");

let port = null;
const RECONNECT_INTERVAL = 500; // ms

function connectWebSocket() {
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);

  ws.addEventListener("open", () => {
    wsStatusEl.textContent = "connected";
    ws.send("ping");
  });

  ws.addEventListener("message", (event) => {
    echoEl.textContent = event.data;
  });

  ws.addEventListener("close", () => {
    wsStatusEl.textContent = "reconnecting…";
    setTimeout(connectWebSocket, RECONNECT_INTERVAL);
  });

  // "error" always fires before "close", so the close handler drives reconnect.
  // We register this listener only to suppress the browser's default console error.
  ws.addEventListener("error", () => {});
}

async function main() {
  port = await invoke("get_backend_port");
  backendStatusEl.textContent = `port ${port}`;
  connectWebSocket();
}

main();
