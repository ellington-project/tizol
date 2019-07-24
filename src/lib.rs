//! A small library for reliable, fast spectrogram generation and visualisation
//!
//! Tizol provides a `Spectrogram` object that allows the user to create spectrograms from raw audio data or audio files, and visualise said spectrograms. The computed spectrogram and visualised images are intended to be as-close-to identical to those computed by [librosa](https://librosa.github.io/librosa/), and should be a drop in replacement for them.
//!
//! # Computing spectrograms
//!
//! Tizol's computation is based heavily off librosa - it uses a simple short time fourier transform (STFT) to generate a raw spectrogram, which is then converted from an amplitude representation to a power representation, before finally being normalised so that elements of the spectrogram fall in the [0,1] range.
//!
//! In librosa terms (omitting the final normalisation), the implementation of `from_file` is roughly as follows:
//! ```python
//! samplerate = 44100
//! (y, sr) = librosa.load(filename, sr=samplerate, res_type='kaiser_best')
//! S = librosa.stft(y)
//! M = librosa.core.magphase(S)[0] # This is implicitly done by the STFT
//! spectrogram = librosa.amplitude_to_db(M, ref=np.max)[0:1024, :]
//! ```
//!
//! # Visualising spectrograms
//!
//! Tizol provides the `Spect::as_image()` method for visualising already-computed spectrograms. This method is unfortunately quite slow as in order to maintain parity with the output of librosa, it uses the `Magma` colourmap from the scarlet crate to compute pixel colours. For some reason, this computation is very slow, and even with parallelisation it is still roughly 10x slower than computing the actual spectrogram.
//!
//! # Protobuf support
//!
//! Spectrogram's support protobuffers through the prost crate, meaning that spectrograms implement the `Message` trait.
//!
//! # Naming
//!
//! Tizol is part of the "Ellington" project - a set of tools designed to make it easier for swing dance DJ's to automatically calculate the tempo of swing music. Each component of the project is named after a member of (or arranger for) Duke Ellington's band. Tizol is named after [Juan Tizol](https://en.wikipedia.org/wiki/Juan_Tizol), a solid rock of the trombone section, and the composer of "Caravan", one of the most famous jazz standards.

// extern crate stft;
mod stft;
use stft::{WindowType, STFT};

extern crate image;
use image::{ImageBuffer, RgbImage};

extern crate hodges;
use hodges::*;

extern crate scarlet;
use crate::scarlet::colormap::ColorMap;

use rayon::prelude::*;

use std::path::PathBuf;

extern crate prost;

include!(concat!(env!("OUT_DIR"), "/tizol.rs"));

impl Spectrogram {
    /// Creates a spectrogram object from a filepath.
    ///
    /// Under the hood, this calls `Spectrogram::from_buffer(...)` with samples read using libhodges.
    ///
    /// Returns `None` if `State::from_file()` fails for any reason.
    pub fn from_file<P: Into<PathBuf>>(filename: P) -> Option<Self> {
        // Create a hodges state object to load the audio data
        let state: State<f32> = State::from_file(filename)?;

        // Collect the audio samples into a single buffer for processing.
        let audio_samples: Vec<f64> = state.map(|f| f as f64).collect();

        Some(Self::from_buffer(&audio_samples))
    }

    /// Creates a spectrogram object from a vector of PCM encoded floating point samples.
    ///
    /// In FFMPEG terms, these are single channel f32le samples, at a sample rate of 44100hz
    pub fn from_buffer(audio_samples: &Vec<f64>) -> Self {
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
        (&audio_samples[..]).chunks(4096).for_each(|samples| {
            stft.append_samples(samples);
            while stft.contains_enough_to_compute() {
                stft.compute_magnitude_column(&mut spectrogram_column[..]);
                spectrogram_output.extend(&spectrogram_column[low_freq..hi_freq]);
                stft.move_to_next_column();
            }
        });

        // Compute the amplitude_to_db of the result.
        Self::amplitude_to_db(&mut spectrogram_output[..]);

        // Normalize the output.
        Self::normalize_buffer_inplace(&mut spectrogram_output[..]);

        // Finally, calculate the height/width of the data.
        let height = hi_freq - low_freq;
        let width = spectrogram_output.len() / height; // Guaranteed to be divisible

        Spectrogram {
            data: spectrogram_output,
            width: width as u32,
            height: height as u32,
        }
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
        buffer
            .iter_mut()
            .for_each(|v| *v = 1.0 - ((v.abs() - min) / range));
    }

    fn amplitude_to_db(s: &mut [f64]) -> () {
        let amin: f64 = 1e-5;
        let top_db: f64 = 80.0;

        let ref_value: f64 =
            s.iter()
                .map(|x| x.abs())
                .fold(std::f64::MIN, |m, x| if x > m { x } else { m });

        // Don't forget to elementwise square S first!
        s.iter_mut().for_each(|v| {
            // Calculate magnitude
            *v = v.abs();
            // Calculate power
            *v = *v * *v;
        });
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

        s.iter_mut().for_each(|v| {
            // Calculate log_spec
            *v = 10.0 * amin.max(*v).log10();
            *v -= 10.0 * amin.max(ref_value).log10();
        });

        let max: f64 = s
            .iter()
            .fold(std::f64::MIN, |m, x| if *x > m { *x } else { m });

        // Second loop after we have the max
        s.iter_mut().for_each(|v| *v = v.max(max - top_db));
    }

    /// Generates an image::Result from a spectrogram.
    ///
    /// This method is significantly slower than (e.g.) `from_buffer`, as it uses the scarlet magma ListedColorMap to compute the output colour in order to maintain parity with librosa/matplotlib
    pub fn as_image(&self) -> RgbImage {
        let mut img: RgbImage = ImageBuffer::new(self.width as u32, self.height as u32);

        let colourmap = scarlet::colormap::ListedColorMap::magma();

        let calculated_colours: Vec<scarlet::color::RGBColor> = self
            .data
            .par_iter()
            .map(|n| colourmap.transform_single(*n))
            .collect();

        calculated_colours[..]
            .chunks(self.height as usize)
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
