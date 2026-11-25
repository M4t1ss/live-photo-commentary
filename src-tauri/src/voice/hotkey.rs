//! The push-to-talk hotkey: a global shortcut, which works while the app is
//! not in focus, reported to the controller as a press and a release.

use tauri::AppHandle;
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

use crate::config::Config;

use super::controller::Input;

/// The key to register for `cfg`: the push-to-talk key, in push-to-talk mode
/// only (the other modes don't use it, and a registered key is taken from
/// every other program).
pub fn effective(cfg: &Config) -> Option<String> {
    let key = cfg.voice_hotkey.as_deref()?.trim();
    (cfg.voice_activation == "push_to_talk" && !key.is_empty()).then(|| key.to_string())
}

/// The plugin, with the handler that turns the key's events into controller
/// inputs. Which key it listens to is set with [`apply`].
pub fn plugin<F>(send: F) -> tauri::plugin::TauriPlugin<tauri::Wry>
where
    F: Fn(&AppHandle, Input) + Send + Sync + 'static,
{
    tauri_plugin_global_shortcut::Builder::new()
        .with_handler(move |app, _shortcut, event| {
            send(
                app,
                match event.state() {
                    ShortcutState::Pressed => Input::KeyPressed,
                    ShortcutState::Released => Input::KeyReleased,
                },
            );
        })
        .build()
}

/// Stops listening to `old` and starts listening to `new`. Fails, leaving
/// no key registered, if `new` isn't a key combination (`Ctrl+Shift+Space`,
/// `F9`) or another program has it.
pub fn apply(app: &AppHandle, old: Option<&str>, new: Option<&str>) -> Result<(), String> {
    if old == new {
        return Ok(());
    }
    let shortcuts = app.global_shortcut();
    if let Some(old) = old.and_then(|key| key.parse::<Shortcut>().ok()) {
        let _ = shortcuts.unregister(old);
    }
    if let Some(new) = new {
        let shortcut: Shortcut = new.parse().map_err(|e| format!("{new:?} is not a key combination: {e}"))?;
        shortcuts.register(shortcut).map_err(|e| format!("cannot use {new:?} as the hotkey: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(activation: &str, hotkey: Option<&str>) -> Config {
        Config { voice_activation: activation.to_string(), voice_hotkey: hotkey.map(str::to_string), ..Config::default() }
    }

    #[test]
    fn the_key_is_registered_only_in_push_to_talk_mode() {
        assert_eq!(effective(&config("push_to_talk", Some("Ctrl+Shift+Space"))), Some("Ctrl+Shift+Space".to_string()));
        assert_eq!(effective(&config("threshold", Some("F9"))), None);
        assert_eq!(effective(&config("off", Some("F9"))), None);
    }

    #[test]
    fn no_key_or_a_blank_one_is_no_key() {
        assert_eq!(effective(&config("push_to_talk", None)), None);
        assert_eq!(effective(&config("push_to_talk", Some("  "))), None);
        assert_eq!(effective(&config("push_to_talk", Some(" F9 "))), Some("F9".to_string()));
    }

    #[test]
    fn the_shortcut_parser_takes_what_the_settings_will_send() {
        for key in ["F9", "F24", "Ctrl+Shift+Space", "Alt+KeyV", "CommandOrControl+Shift+KeyM"] {
            assert!(key.parse::<Shortcut>().is_ok(), "{key}");
        }
        assert!("not a key".parse::<Shortcut>().is_err());
    }
}
