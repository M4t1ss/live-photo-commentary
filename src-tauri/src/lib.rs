use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

struct BackendState {
    port: u16,
    #[allow(dead_code)] // held for Drop, which kills the child process on exit
    process: BackendProcess,
}

struct BackendProcess(Mutex<Option<Child>>);

impl Drop for BackendProcess {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.0.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

#[tauri::command]
fn get_backend_port(state: State<BackendState>) -> u16 {
    state.port
}

fn find_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

async fn poll_until_ready(port: u16) {
    let url = format!("http://127.0.0.1:{}/health", port);
    let client = reqwest::Client::new();
    loop {
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = find_free_port();

    // In dev, CARGO_MANIFEST_DIR is src-tauri/; parent is the project root.
    // TODO: use bundled resources path in production.
    let backend_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has no parent")
        .join("backend");

    let port_str = port.to_string();
    let child = Command::new("uv")
        .args(["run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port"])
        .arg(&port_str)
        .current_dir(&backend_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to spawn backend — is `uv` in PATH?");

    tauri::Builder::default()
        .setup(move |app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            app.manage(BackendState {
                port,
                process: BackendProcess(Mutex::new(Some(child))),
            });

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                poll_until_ready(port).await;
                let _ = handle.emit("backend_ready", port);
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_backend_port])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
