extern crate tizol;

extern crate stft;
use stft::{WindowType, STFT};

extern crate image;
use image::{ImageBuffer, RgbImage};

extern crate hodges;
use hodges::*;

extern crate scarlet;
use crate::scarlet::colormap::ColorMap;

use std::time::{Duration, Instant};

fn main() {
    let start = Instant::now();
    println!("==== Time: {:?}", start.elapsed());
    println!("Reading file");

    let audio_f = "One O'Clock Jump - Metronome All Star Band.mp3";

    let state: State<f32> = State::from_file(audio_f).expect("Failed to open file with libhodges");

    let audio_samples: Vec<f64> = state.map(|f| f as f64).collect();

    // Dump the audio samples

    println!("==== Time: {:?}", start.elapsed());
    println!("Collected samples!");
    println!("Found {} samples", audio_samples.len());

    // let's initialize our short-time fourier transform
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 2048;
    let step_size: usize = window_size / 4;
    let mut stft = STFT::<f64>::new(window_type, window_size, step_size);

    // we need a buffer to hold a computed column of the spectrogram
    let mut spectrogram_column: Vec<f64> = std::iter::repeat(0.).take(stft.output_size()).collect();

    let mut raw_spectrogram_data: Vec<f64> =
        Vec::with_capacity(stft.output_size() * audio_samples.len() / window_size);

    let low_freq = 0;
    let hi_freq = 1024;

    // iterate over all the samples in chunks of 3000 samples.
    // in a real program you would probably read from something instead.
    for some_samples in (&audio_samples[..]).chunks(4096) {
        // append the samples to the internal ringbuffer of the stft
        stft.append_samples(some_samples);

        // as long as there remain window_size samples in the internal
        // ringbuffer of the stft
        while stft.contains_enough_to_compute() {
            stft.compute_magnitude_column(&mut spectrogram_column[..]);
            raw_spectrogram_data.extend(&spectrogram_column[low_freq..hi_freq]);
            stft.move_to_next_column();
        }
    }

    // let raw_spectrogram_data : Vec<f64> = tizol::util::read_sample_file("python_samples.txt");

    tizol::amplitude_to_db(&mut raw_spectrogram_data[..]);
    println!("==== Time: {:?}", start.elapsed());
    println!(
        "{} elements in raw_spectrogram_data",
        raw_spectrogram_data.len()
    );
    println!(
        "{} columns in raw_spectrogram_data",
        raw_spectrogram_data.len() / stft.output_size()
    );

    // normalise the vector
    let (min, max): (f64, f64) = raw_spectrogram_data.iter().map(|v| v .abs() ).fold(
        (std::f64::MAX, std::f64::MIN),
        |(mi, ma), x| {
            let new_mi = if x < mi { x } else { mi };
            let new_ma = if x > ma { x } else { ma };
            (new_mi, new_ma)
        },
    );

    let range = max - min;

    println!("Range {} min {} max {}", range, min, max);
    let normalized: Vec<f64> = raw_spectrogram_data
        .iter()
        .map(|v| 1.0 - ((v.abs() - min) / range))
        .collect();

    println!("==== Time: {:?}", start.elapsed());
    println!("Normalized spectrogram data");

    let image_height: usize = hi_freq - low_freq; //stft.output_size() as usize;
    let image_width: usize = 1 + (raw_spectrogram_data.len() / image_height);

    let mut img: RgbImage = ImageBuffer::new(image_width as u32, image_height as u32);

    let colourmap = scarlet::colormap::ListedColorMap::magma();

    let calculated_colours : Vec<scarlet::color::RGBColor>  = colourmap.transform(normalized);

    println!("==== Time: {:?}", start.elapsed());
    println!("Calculated colours");

    for (c, column) in calculated_colours[..].chunks(image_height).enumerate() {
        for (r, colour) in column.iter().enumerate() {
            // let colour: scarlet::color::RGBColor = colourmap.transform_single(1.0 - spect_dat);
            let p = image::Rgb([colour.int_r(), colour.int_g(), colour.int_b()]);
            img.put_pixel(c as u32, image_height as u32 - r as u32 - 1, p);
        }
    }

    img.save("test.png").unwrap();
    println!("==== Time: {:?}", start.elapsed());
    println!("Finished.");
}
