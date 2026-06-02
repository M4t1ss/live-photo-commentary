use screenshots::Screen;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: screenshot <output.png>");
        std::process::exit(1);
    }

    let screens = Screen::all().unwrap_or_else(|e| {
        eprintln!("Failed to enumerate screens: {e}");
        std::process::exit(1);
    });

    let screen = screens.first().unwrap_or_else(|| {
        eprintln!("No screens found");
        std::process::exit(1);
    });

    screen
        .capture()
        .unwrap_or_else(|e| {
            eprintln!("Capture failed: {e}");
            std::process::exit(1);
        })
        .save(&args[1])
        .unwrap_or_else(|e| {
            eprintln!("Save failed: {e}");
            std::process::exit(1);
        });
}
