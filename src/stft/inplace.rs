/*!
 *
 * DERIVED HEAVILY FROM THE STFT REPO ON GITHUB

**computes the [short-time fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
on inplace data, in parallel**

*/
use super::*;
use rayon::prelude::*;

pub struct STFT<T>
where
    T: FFTnum + FromF64 + num::Float,
{
    pub window_size: usize,
    pub step_size: usize,
    pub fft: Arc<dyn FFT<T>>,
    pub window: Option<Vec<T>>,
}

impl<T> STFT<T>
where
    T: FFTnum + FromF64 + num::Float + std::fmt::Display,
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
            window: window,
        }
    }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.window_size / 2
    }

    #[inline]
    pub fn stft_size(&self, input_length: usize) -> usize {
        // Get the number of strides that we potentially have in the input
        // Essentially, subtract the "final" window size, and then work out how
        // many strides we have left in the remaining input (which isn't dependent on the window size)
        let strides: usize = (input_length - self.window_size) / self.step_size;
        // Work out the total output size based on strides * window_size
        strides * self.window_size
    }

    pub fn compute_complex_column(&self, real_input: &[T]) -> Vec<Complex<T>> {
        assert!(self.window_size <= real_input.len());

        // multiply real_input with window, and make complex
        let mut complex_input: Vec<Complex<T>> = if let Some(ref window) = self.window {
            real_input
                .iter()
                .zip(window.iter())
                .map(|(i, w)| (*i * *w).into())
                .collect()
        } else {
            real_input.iter().map(|i| (*i).into()).collect()
        };

        let mut complex_output: Vec<Complex<T>> = std::iter::repeat(Complex::<T>::zero())
            .take(self.window_size)
            .collect();
        //  compute fft
        self.fft.process(&mut complex_input, &mut complex_output);

        complex_output
    }

    pub fn compute_magnitude_column(&self, real_input: &[T]) -> Vec<T> {
        self.compute_complex_column(real_input)
            .iter()
            .take(self.output_size())
            .map(|elem| elem.norm())
            .collect()
    }

    pub fn compute_column(&self, real_input: &[T]) -> Vec<T> {
        self.compute_complex_column(real_input)
            .par_iter()
            .take(self.window_size/2)
            .map(|elem| log10_positive(elem.norm()))
            .collect()
    }

    // Hardcode all this into a single function for now.
    pub fn stft(&self, data: &Vec<T>) -> Vec<T> {
        // for(
        // usize window_start_ix = 0;
        // window_start_ix < data.len() - self.window_size;
        // window_start_ix += self.step_size)

        let mut result_vec: Vec<T> = vec![];

        let mut window_start_ix = 0;
        while window_start_ix < data.len() - self.window_size {
            let window_end_ix = window_start_ix + self.window_size;
            result_vec.extend(self.compute_magnitude_column(&data[window_start_ix..window_end_ix]));
            window_start_ix += self.step_size;
        }
        result_vec
    }

    pub fn iter_stft(&self, data: &Vec<T>) -> Vec<T> {
        data[..]
            .windows(self.window_size)
            .step_by(self.step_size)
            .map(|window| self.compute_magnitude_column(window))
            .flatten()
            .collect()
    }

    pub fn par_iter_stft(&self, data: &Vec<T>) -> Vec<T> {
        data[..]
            .par_windows(self.window_size)
            .step_by(self.step_size)
            .map(|window| self.compute_magnitude_column(window))
            .flatten()
            .collect()
    }
}
