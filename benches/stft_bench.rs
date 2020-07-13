use criterion::{criterion_group, criterion_main, Criterion};
use tizol::stft::inplace::STFT as ISTFT;
use tizol::stft::streaming::STFT as SSTFT;
use tizol::stft::WindowType;

fn generate_samples() -> std::vec::Vec<f64> {
    let sample_rate: usize = 44100;
    let seconds: usize = 10;
    let sample_count = sample_rate * seconds;
    let all_samples = (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>();
    all_samples
}

fn streaming_stft(all_samples: &std::vec::Vec<f64>) {
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
}

fn inplace_stft(all_samples: &std::vec::Vec<f64>) {
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 1024;
    let step_size: usize = 512;

    let istft = ISTFT::new(window_type, window_size, step_size);
    let _inplace_result = istft.stft(&all_samples);
}

fn iter_inplace_stft(all_samples: &std::vec::Vec<f64>) {
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 1024;
    let step_size: usize = 512;

    let istft = ISTFT::new(window_type, window_size, step_size);
    let _inplace_result = istft.iter_stft(&all_samples);
}

fn par_iter_inplace_stft(all_samples: &std::vec::Vec<f64>) {
    let window_type: WindowType = WindowType::Hanning;
    let window_size: usize = 1024;
    let step_size: usize = 512;

    let istft = ISTFT::new(window_type, window_size, step_size);
    let _inplace_result = istft.par_iter_stft(&all_samples);
}

fn streaming_bench(c: &mut Criterion) {
    let samples = generate_samples();
    c.bench_function("streaming stft", |b| b.iter(|| streaming_stft(&samples)));
}

fn inplace_bench(c: &mut Criterion) {
    let samples = generate_samples();
    c.bench_function("inplace stft", |b| b.iter(|| inplace_stft(&samples)));
}

fn iter_inplace_bench(c: &mut Criterion) {
    let samples = generate_samples();
    c.bench_function("iter_inplace stft", |b| {
        b.iter(|| iter_inplace_stft(&samples))
    });
}

fn par_iter_inplace_bench(c: &mut Criterion) {
    let samples = generate_samples();
    c.bench_function("par_iter_inplace stft", |b| {
        b.iter(|| par_iter_inplace_stft(&samples))
    });
}

criterion_group!(
    benches,
    streaming_bench,
    inplace_bench,
    iter_inplace_bench,
    par_iter_inplace_bench
);
criterion_main!(benches);
