extern crate tizol;
use tizol::Spectrogram;


use std::time::Instant;



fn main() {
    let start = Instant::now();

    let log_time = |message: &'static str| -> () {
        println!("==== Time: {: >15?} ==== {}", start.elapsed(), message);
    };

    log_time("Read file");

    let audio_f = "One O'Clock Jump - Metronome All Star Band.mp3";

    let sp = Spectrogram::from_file(audio_f).unwrap();

    log_time("Computed spectrogram");

    let img = sp.as_image();

    log_time("Image generated!");

    img.save("test.png").unwrap();

    log_time("Finished.");
}
