use screenshots::Screen;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--list-monitors") {
        list_monitors();
        return;
    }

    let output = args.first().unwrap_or_else(|| {
        eprintln!("Usage: screenshot <output.png> [--monitor <idx>] [--clip x,y,w,h]");
        eprintln!("       screenshot --list-monitors");
        std::process::exit(1);
    });

    let monitor_idx = parse_flag_usize(&args, "--monitor").unwrap_or(0);
    let clip = parse_flag_clip(&args, "--clip");

    let screens = Screen::all().unwrap_or_else(|e| {
        eprintln!("Failed to enumerate screens: {e}");
        std::process::exit(1);
    });

    let screen = screens.get(monitor_idx).unwrap_or_else(|| {
        eprintln!("Monitor index {monitor_idx} not found ({} monitors available)", screens.len());
        std::process::exit(1);
    });

    let mut img = screen.capture().unwrap_or_else(|e| {
        eprintln!("Capture failed: {e}");
        std::process::exit(1);
    });

    if let Some([x, y, w, h]) = clip {
        let ix = x.max(0) as u32;
        let iy = y.max(0) as u32;
        let iw = (w as u32).min(img.width().saturating_sub(ix)).max(1);
        let ih = (h as u32).min(img.height().saturating_sub(iy)).max(1);
        img = image::DynamicImage::ImageRgba8(img).crop_imm(ix, iy, iw, ih).into_rgba8();
    }

    img.save(output).unwrap_or_else(|e| {
        eprintln!("Save failed: {e}");
        std::process::exit(1);
    });
}

fn list_monitors() {
    let screens = Screen::all().unwrap_or_else(|e| {
        eprintln!("Failed to enumerate screens: {e}");
        std::process::exit(1);
    });
    for (i, s) in screens.iter().enumerate() {
        let d = &s.display_info;
        println!(
            "{}: Display {} {}x{}@({},{}){}",
            i,
            d.id,
            d.width,
            d.height,
            d.x,
            d.y,
            if d.is_primary { " [primary]" } else { "" },
        );
    }
}

fn parse_flag_usize(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}

fn parse_flag_clip(args: &[String], flag: &str) -> Option<[i32; 4]> {
    let val = args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())?;
    let parts: Vec<i32> = val.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if parts.len() == 4 {
        Some([parts[0], parts[1], parts[2], parts[3]])
    } else {
        None
    }
}
