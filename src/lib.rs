// extern crate itertools;
// extern crate itertools_num;

// use itertools_num::linspace;
// use itertools_num::Linspace;

pub fn generate_random_samples(sample_rate: usize, seconds: usize) -> Vec<f64> {
    let sample_count = sample_rate * seconds;
    (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>()
}

pub fn amplitude_to_db(S: &mut [f64]) -> () {
    let amin: f64 = 1e-5;
    let top_db: f64 = 80.0;

    let ref_value: f64 = S
        .iter()
        .map(|x| x.abs())
        .fold(std::f64::MIN, |m, x| if x > m { x } else { m });

    // Don't forget to elementwise square S first!
    for v in S.iter_mut() {
        // Calculate magnitude
        *v = v.abs();
        // Calculate power
        *v = *v * *v;
    }
    power_to_db(S, ref_value.powf(2.0), amin.powf(2.0), top_db);
}

pub fn power_to_db(S: &mut [f64], ref_value: f64, amin: f64, top_db: f64) -> () {
    if amin <= 0.0 {
        panic!("amin must be >= 0");
    }
    if top_db < 0.0 {
        panic!("top_db must be > 0");
    }

    let ref_value = ref_value.abs();

    for v in S.iter_mut() {
        // Calculate log_spec
        *v = 10.0 * amin.max(*v).log10();
        *v -= 10.0 * amin.max(ref_value).log10();
    }

    let max: f64 = S
        .iter()
        .fold(std::f64::MIN, |m, x| if *x > m { *x } else { m });

    // Second loop after we have the max
    for v in S.iter_mut() {
        *v = v.max(max - top_db);
    }
}

// default: sr=22050, n_fft=2048
// pub fn fft_frequencies(sr: u32, n_fft: u32) -> Linspace<f64> {
//     linspace::<f64>(0.0, sr as f64 / 2.0, (1 + n_fft / 2) as usize)
// }

// def __coord_fft_hz(n, sr=22050, **_kwargs):
//     '''Get the frequencies for FFT bins'''
//     n_fft = 2 * (n - 1)
//     # The following code centers the FFT bins at their frequencies
//     # and clips to the non-negative frequency range [0, nyquist]
//     basis = core.fft_frequencies(sr=sr, n_fft=n_fft)
//     fmax = basis[-1]
//     basis -= 0.5 * (basis[1] - basis[0])
//     basis = np.append(np.maximum(0, basis), [fmax])
//     return basis
