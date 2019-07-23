extern crate tizol;
use tizol::Spectrogram;

use std::time::Instant;

use std::env;

fn main() {

    let args: Vec<String> = env::args().collect();
    let in_f = args[1].clone();
    let out_f = args[2].clone();

    let start = Instant::now();

    let log_time = |message: &'static str| -> () {
        println!("==== Time: {: >15?} ==== {}", start.elapsed(), message);
    };

    log_time("Read file");

    let sp = Spectrogram::from_file(in_f).unwrap();

    log_time("Computed spectrogram");

    let img = sp.as_image();

    log_time("Image generated!");

    img.save(out_f).unwrap();

    log_time("Finished.");
}
