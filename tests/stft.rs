use tizol::stft::inplace::STFT as ISTFT;
use tizol::stft::streaming::STFT as SSTFT;
use tizol::stft::WindowType;

#[cfg(test)]
#[test]
fn complete_stft() {
    // ten seconds of generated fake audio
    let sample_rate: usize = 44100;
    let seconds: usize = 10;
    let sample_count = sample_rate * seconds;
    let all_samples = (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>();

    // let's initialize our short-time fourier transform
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 1024;
    let step_size: usize = 512;

    // initialise streaming short time fourier transform
    let mut sstft = SSTFT::new(window_type, window_size, step_size);

    let mut spectrogram_column: Vec<f64> =
        std::iter::repeat(0.).take(sstft.output_size()).collect();

    let mut reference_vector = vec![];

    for some_samples in (&all_samples[..]).chunks(3000) {
        sstft.append_samples(some_samples);

        while sstft.contains_enough_to_compute() {
            sstft.compute_column(&mut spectrogram_column[..]);
            reference_vector.extend(spectrogram_column.clone());
            sstft.move_to_next_column();
        }
    }

    // Compute an inplace fourier transform
    let istft = ISTFT::new(window_type, window_size, step_size);
    let inplace_result = istft.stft(&all_samples);
    let iter_result = istft.iter_stft(&all_samples);

    println!("Reference result length: {:?}", reference_vector.len());
    println!("Inplace result length: {:?}", inplace_result.len());

    assert_eq!(reference_vector, inplace_result);
    assert_eq!(reference_vector, iter_result);
}
