pub fn something() -> i32 {
    return 0;
}

pub fn generate_random_samples(sample_rate: usize, seconds: usize) -> Vec<f64> {
    let sample_count = sample_rate * seconds;
    (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>()
}

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
