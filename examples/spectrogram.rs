extern crate tizol;
use tizol::Spectrogram;

use std::time::Instant;

use std::env;

use prost::Message;
use std::fs::File;
use std::io::{Read, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    let in_f = args[1].clone();
    let out_f = args[2].clone();

    // let start = Instant::now();

    // let log_time = |message: &'static str| -> () {
    //     println!("==== Time: {: >15?} ==== {}", start.elapsed(), message);
    // };

    // Read an audio file into a spectrogram
    let sp = Spectrogram::from_file(in_f).unwrap();
    // log_time("Computed spectrogram");

    // Save it as an image
    let img = sp.as_image();
    // log_time("Image generated!");

    img.save(out_f).unwrap();
    // log_time("Image saved.");

    // Do some protobuf stuff.
    // let mut buf = Vec::<u8>::new();

    // // Encode it.
    // sp.encode(&mut buf).unwrap();
    // log_time("Encoded.");

    // {
    //     // Save it.
    //     let mut file = File::create("rust.buf").unwrap();
    //     file.write_all(&buf[..]).unwrap();
    //     log_time("Binary file written");
    // }

    // {
    //     // Load it.
    //     let mut file = File::open("rust.buf").unwrap();
    //     let mut buffer = Vec::<u8>::new();
    //     file.read_to_end(&mut buffer).unwrap();
    //     log_time("Binary file read.");
    //     let sp2 = Spectrogram::decode(&buffer).unwrap();
    //     log_time("Binary file decoded.");
    //     let img2 = sp2.as_image();
    //     img2.save("rust_buf.png");
    //     log_time("Second image saved");
    // }
}
