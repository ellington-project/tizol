extern crate tizol;

extern crate stft;
use stft::{WindowType, STFT};

extern crate image;
use image::{GenericImageView, ImageBuffer, RgbImage};

extern crate hodges;
use hodges::*;

extern crate scarlet;
use crate::scarlet::colormap::ColorMap;

fn main() {
    let audio_f = "/home/adam/Music/One O'Clock Jump - Metronome All Star Band.mp3";

    let state: State<f32> = State::from_file(audio_f).expect("Failed to open file with libhodges");

    let audio_samples: Vec<f64> = state.map(|f| f as f64).collect();



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

    // iterate over all the samples in chunks of 3000 samples.
    // in a real program you would probably read from something instead.
    for some_samples in (&audio_samples[..]).chunks(3000) {
        // append the samples to the internal ringbuffer of the stft
        stft.append_samples(some_samples);

        // as long as there remain window_size samples in the internal
        // ringbuffer of the stft
        while stft.contains_enough_to_compute() {
            // compute one column of the stft by
            // taking the first window_size samples of the internal ringbuffer,
            // multiplying them with the window,
            // computing the fast fourier transform,
            // taking half of the symetric complex outputs,
            // computing the norm of the complex outputs and
            // taking the log10
            stft.compute_magnitude_column(&mut spectrogram_column[..]);

            // here's where you would do something with the

            tizol::amplitude_to_db(&mut spectrogram_column[..]);

            raw_spectrogram_data.extend(&spectrogram_column[64..320]);

            // drop step_size samples from the internal ringbuffer of the stft
            // making a step of size step_size
            stft.move_to_next_column();
        }
    }

    println!(
        "{} elements in raw_spectrogram_data",
        raw_spectrogram_data.len()
    );
    println!(
        "{} columns in raw_spectrogram_data",
        raw_spectrogram_data.len() / stft.output_size()
    );

    // normalise the vector
    let (min, max): (f64, f64) = raw_spectrogram_data.iter().map(|v| v.abs()).fold(
        (std::f64::MAX, std::f64::MIN),
        |(mi, ma), x| {
            let new_mi = if x < mi { x } else { mi };
            let new_ma = if x > ma { x } else { ma };
            (new_mi, new_ma)
        },
    );

    let range = max - min;
    let normalized: Vec<f64> = raw_spectrogram_data
        .iter()
        .map(|v| (v.abs() - min) / range)
        .collect();

    let image_height: usize = 320 - 64; //stft.output_size() as usize;
    let image_width: usize = 1 + (raw_spectrogram_data.len() / image_height);

    let mut img: RgbImage = ImageBuffer::new(image_width as u32, image_height as u32);

    let colourmap = scarlet::colormap::ListedColorMap::magma();
    let normalizer = scarlet::colormap::NormalizeMapping::Cbrt;

    for (c, column) in normalized[..].chunks(image_height).enumerate() {
        for (r, spect_dat) in column.iter().enumerate() {
            let norm_dat = normalizer.normalize(*spect_dat);
            let colour: scarlet::color::RGBColor = colourmap.transform_single(1.0 - norm_dat);
            let p = image::Rgb([colour.int_r(), colour.int_g(), colour.int_b()]);
            img.put_pixel(c as u32, image_height as u32 - r as u32 - 1, p);
        }
    }

    img.save("test.png").unwrap();
}
