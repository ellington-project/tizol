extern crate stft;
use stft::{WindowType, STFT};

extern crate image;
use image::{ImageBuffer, RgbImage};

extern crate hodges;
use hodges::*;

extern crate scarlet;
use crate::scarlet::colormap::ColorMap;

use std::time::Instant;

use rayon::prelude::*;

use std::fs;

use std::path::{Path, PathBuf};

pub struct Spectrogram {
    pub data: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl Spectrogram {
    /* Open an audio file, read it into a buffer, then call `from_buffer` */
    pub fn from_file<P: Into<PathBuf>>(filename: P) -> Option<Self> {
        // Create a hodges state object to load the audio data
        let state: State<f32> = match State::from_file(filename) {
            Some(s) => s,
            None => {
                println!("Failed to open file with libhodges");
                return None;
            }
        };

        // Collect the audio samples into a single buffer for processing.
        let audio_samples: Vec<f64> = state.map(|f| f as f64).collect();

        Self::from_buffer(&audio_samples)
    }

    pub fn from_buffer(audio_samples: &Vec<f64>) -> Option<Self> {
        // Initialise the stft machinery.
        let mut stft = STFT::<f64>::new(
            WindowType::Hanning, /*The STFT window type*/
            2048,                /* Window size */
            2048 / 4,            /* Step size */
        );

        /* Create buffers for:
            - The output of a single STFT column,
            - The total output of the STFT
        */
        let mut spectrogram_column: Vec<f64> =
            std::iter::repeat(0.).take(stft.output_size()).collect();

        let mut spectrogram_output: Vec<f64> = Vec::with_capacity(audio_samples.len() * 2);

        // Adjustable - for now, use the whole buffer.
        let low_freq = 0;
        let hi_freq = 1024;

        // Perform the STFT across the samples
        for samples in (&audio_samples[..]).chunks(4096) {
            stft.append_samples(samples);
            while stft.contains_enough_to_compute() {
                stft.compute_magnitude_column(&mut spectrogram_column[..]);
                spectrogram_output.extend(&spectrogram_column[low_freq..hi_freq]);
                stft.move_to_next_column();
            }
        }

        // Compute the amplitude_to_db of the result.
        Self::amplitude_to_db(&mut spectrogram_output[..]);

        // Normalize the output.
        Self::normalize_buffer_inplace(&mut spectrogram_output[..]);

        // Finally, calculate the height/width of the data.
        let height = hi_freq - low_freq;
        let width = (spectrogram_output.len() / height); // Guaranteed to be divisible

        Some(Spectrogram {
            data: spectrogram_output,
            width: width,
            height: height,
        })
    }

    fn normalize_buffer_inplace(buffer: &mut [f64]) -> () {
        let (min, max): (f64, f64) =
            buffer
                .iter()
                .map(|v| v.abs())
                .fold((std::f64::MAX, std::f64::MIN), |(mi, ma), x| {
                    let new_mi = if x < mi { x } else { mi };
                    let new_ma = if x > ma { x } else { ma };
                    (new_mi, new_ma)
                });

        let range = max - min;
        for v in buffer.iter_mut() {
            *v = 1.0 - ((v.abs() - min) / range);
        }
    }

    fn amplitude_to_db(s: &mut [f64]) -> () {
        let amin: f64 = 1e-5;
        let top_db: f64 = 80.0;

        let ref_value: f64 =
            s.iter()
                .map(|x| x.abs())
                .fold(std::f64::MIN, |m, x| if x > m { x } else { m });

        // Don't forget to elementwise square S first!
        for v in s.iter_mut() {
            // Calculate magnitude
            *v = v.abs();
            // Calculate power
            *v = *v * *v;
        }
        Self::power_to_db(s, ref_value.powf(2.0), amin.powf(2.0), top_db);
    }

    fn power_to_db(s: &mut [f64], ref_value: f64, amin: f64, top_db: f64) -> () {
        if amin <= 0.0 {
            panic!("amin must be >= 0");
        }
        if top_db < 0.0 {
            panic!("top_db must be > 0");
        }

        let ref_value = ref_value.abs();

        for v in s.iter_mut() {
            // Calculate log_spec
            *v = 10.0 * amin.max(*v).log10();
            *v -= 10.0 * amin.max(ref_value).log10();
        }

        let max: f64 = s
            .iter()
            .fold(std::f64::MIN, |m, x| if *x > m { *x } else { m });

        // Second loop after we have the max
        for v in s.iter_mut() {
            *v = v.max(max - top_db);
        }
    }

    pub fn as_image(&self) -> RgbImage {
        let mut img: RgbImage = ImageBuffer::new(self.width as u32, self.height as u32);

        let colourmap = scarlet::colormap::ListedColorMap::magma();

        let calculated_colours: Vec<scarlet::color::RGBColor> = self
            .data
            .par_iter()
            .map(|n| colourmap.transform_single(*n))
            .collect();

        calculated_colours[..]
            .chunks(self.height)
            .enumerate()
            .for_each(|(c, column)| {
                column.iter().enumerate().for_each(|(r, colour)| {
                    let p = image::Rgb([colour.int_r(), colour.int_g(), colour.int_b()]);
                    img.put_pixel(c as u32, self.height as u32 - r as u32 - 1, p);
                })
            });

        img
    }
}
