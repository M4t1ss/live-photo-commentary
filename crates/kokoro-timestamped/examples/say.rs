//! Speaks text into `output.wav` and prints the timings.
//!
//! Usage: cargo run --example say -- [--voice NAME] [TEXT...]

use kokoro_timestamped::{KokoroSynthesizer, Voice};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Show warnings and errors by default; override with e.g. RUST_LOG=debug.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut voice = "af_heart".to_string();
    if args.first().is_some_and(|a| a == "--voice") && args.len() >= 2 {
        voice = args.remove(1);
        args.remove(0);
    }

    let mut synthesizer = KokoroSynthesizer::new(
        Voice::Name(voice),
        "model",
        None,
        1.0,
    ).await?;

    let text = if args.is_empty() {
        "{happy} Hello, world! I have 123 apples. \
            Please record the record before the object objects. {surprised}"
            .to_string()
    } else {
        args.join(" ")
    };

    // All chunks go into one file, as 16-bit mono for maximum player support.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: KokoroSynthesizer::SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create("output.wav", spec)?;

    for chunk in synthesizer.synthesize(&text)? {
        let chunk = chunk?;
        for &sample in &chunk.audio {
            wav.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        println!("Chunk: {:?} ({} samples)", chunk.text, chunk.audio.len());
        let phonemes: String = chunk.phonemes.iter().map(|(p, _)| p).collect();
        println!("  phonemes: {phonemes}");
        match &chunk.words {
            Some(words) => {
                for word in words {
                    println!(
                        "  {:?} chars {}..{}: {:.3}s - {:.3}s",
                        word.text, word.char_start, word.char_end, word.start, word.end
                    );
                }
            }
            None => println!("  (word alignment failed)"),
        }
        println!("  tags: {:?}", chunk.tags);
    }

    wav.finalize()?;
    println!("Wrote output.wav");
    Ok(())
}
