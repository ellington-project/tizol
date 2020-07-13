/*!
 *
 * COPIED FROM THE STFT REPO ON GITHUB

**computes the [short-time fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
on streaming data.**

to use add `stft = "*"`
to the `[dependencies]` section of your `Cargo.toml` and call `extern crate stft;` in your code.

## example

```ignore
use stft::{WindowType, STFT};

fn main() {
    // let's generate ten seconds of fake audio
    let sample_rate: usize = 44100;
    let seconds: usize = 10;
    let sample_count = sample_rate * seconds;
    let all_samples = (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>();

    // let's initialize our short-time fourier transform
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 1024;
    let step_size: usize = 512;
    let mut stft = STFT::new(window_type, window_size, step_size);

    // we need a buffer to hold a computed column of the spectrogram
    let mut spectrogram_column: Vec<f64> =
        std::iter::repeat(0.).take(stft.output_size()).collect();

    // iterate over all the samples in chunks of 3000 samples.
    // in a real program you would probably read from something instead.
    for some_samples in (&all_samples[..]).chunks(3000) {
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
            stft.compute_column(&mut spectrogram_column[..]);

            // here's where you would do something with the
            // spectrogram_column...

            // drop step_size samples from the internal ringbuffer of the stft
            // making a step of size step_size
            stft.move_to_next_column();
        }
    }
}
```
*/

use super::*;

pub struct STFT<T>
where
    T: FFTnum + FromF64 + num::Float,
{
    pub window_size: usize,
    pub step_size: usize,
    pub fft: Arc<dyn FFT<T>>,
    pub window: Option<Vec<T>>,
    /// internal ringbuffer used to store samples
    pub sample_ring: SliceRingImpl<T>,
    pub real_input: Vec<T>,
    pub complex_input: Vec<Complex<T>>,
    pub complex_output: Vec<Complex<T>>,
}

impl<T> STFT<T>
where
    T: FFTnum + FromF64 + num::Float,
{
    pub fn new(window_type: WindowType, window_size: usize, step_size: usize) -> Self {
        let window = window_type.as_window_vec(window_size);
        Self::new_with_window_vec(window, window_size, step_size)
    }

    // TODO this should ideally take an iterator and not a vec
    pub fn new_with_window_vec(
        window: Option<Vec<T>>,
        window_size: usize,
        step_size: usize,
    ) -> Self {
        // TODO more assertions:
        // window_size is power of two
        // step_size > 0
        assert!(step_size <= window_size);
        let inverse = false;
        let mut planner = FFTplanner::new(inverse);
        STFT {
            window_size: window_size,
            step_size: step_size,
            fft: planner.plan_fft(window_size),
            sample_ring: SliceRingImpl::new(),
            window: window,
            real_input: std::iter::repeat(T::zero()).take(window_size).collect(),
            complex_input: std::iter::repeat(Complex::<T>::zero())
                .take(window_size)
                .collect(),
            complex_output: std::iter::repeat(Complex::<T>::zero())
                .take(window_size)
                .collect(),
        }
    }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.window_size
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.sample_ring.len()
    }

    pub fn append_samples(&mut self, input: &[T]) {
        self.sample_ring.push_many_back(input);
    }

    #[inline]
    pub fn contains_enough_to_compute(&self) -> bool {
        self.window_size <= self.sample_ring.len()
    }

    pub fn compute_into_complex_output(&mut self) {
        assert!(self.contains_enough_to_compute());

        // read into real_input
        self.sample_ring.read_many_front(&mut self.real_input[..]);

        // multiply real_input with window
        if let Some(ref window) = self.window {
            for (dst, src) in self.real_input.iter_mut().zip(window.iter()) {
                *dst = *dst * *src;
            }
        }

        // copy windowed real_input as real parts into complex_input
        for (dst, src) in self.complex_input.iter_mut().zip(self.real_input.iter()) {
            dst.re = src.clone();
        }

        // compute fft
        self.fft
            .process(&mut self.complex_input, &mut self.complex_output);
    }

    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_complex_column(&mut self, output: &mut [Complex<T>]) {
        assert_eq!(self.output_size(), output.len());

        self.compute_into_complex_output();

        for (dst, src) in output.iter_mut().zip(self.complex_output.iter()) {
            *dst = src.clone();
        }
    }

    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_magnitude_column(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.compute_into_complex_output();

        for (dst, src) in output.iter_mut().zip(self.complex_output.iter()) {
            *dst = src.norm();
        }
    }

    /// computes a column of the spectrogram
    /// # Panics
    /// panics unless `self.output_size() == output.len()`
    pub fn compute_column(&mut self, output: &mut [T]) {
        assert_eq!(self.output_size(), output.len());

        self.compute_into_complex_output();

        for (dst, src) in output.iter_mut().zip(self.complex_output.iter()) {
            *dst = log10_positive(src.norm());
        }
    }

    /// make a step
    /// drops `self.step_size` samples from the internal buffer `self.sample_ring`.
    pub fn move_to_next_column(&mut self) {
        self.sample_ring.drop_many_front(self.step_size);
    }
}
