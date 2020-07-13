extern crate tizol;
use tizol::Spectrogram;

use std::time::Instant;

// use std::env;

fn main() {
    // let args: Vec<String> = env::args().collect();
    let in_f = "ocj.mp3"; //args[1].clone();
    let iterations = 10; //args[2].clone().parse::<u32>().unwrap();

    let t0 = Instant::now();

    let log_time = |iter: u32| -> () {
        println!(
            "Current progress: {} / {}s / {}s",
            iter,
            t0.elapsed().as_secs_f32(),
            t0.elapsed().as_secs_f32() / (iter as f32 + 1.0)
        );
    };

    for i in 0..iterations {
        let sp = Spectrogram::from_file(&in_f).unwrap();

        let img = sp.as_image_bw_raw();

        println!("Width: {}, Height: {}", img.width(), img.height());
        log_time(i);
    }
}
