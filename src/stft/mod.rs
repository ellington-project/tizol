pub mod inplace;
pub mod streaming;

use std::str::FromStr;
use std::sync::Arc;

extern crate num;
use num::complex::Complex;
use num::traits::{Float, Signed, Zero};

extern crate apodize;

extern crate strider;
use strider::{SliceRing, SliceRingImpl};

extern crate rustfft;
use rustfft::{FFTnum, FFTplanner, FFT};

/// the type of apodization window to use
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub enum WindowType {
    Hanning,
    Hamming,
    Blackman,
    Nuttall,
    None,
}

impl FromStr for WindowType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        match &lower[..] {
            "hanning" => Ok(WindowType::Hanning),
            "hann" => Ok(WindowType::Hanning),
            "hamming" => Ok(WindowType::Hamming),
            "blackman" => Ok(WindowType::Blackman),
            "nuttall" => Ok(WindowType::Nuttall),
            "none" => Ok(WindowType::None),
            _ => Err("no match"),
        }
    }
}

// this also implements ToString::to_string
impl std::fmt::Display for WindowType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{:?}", self)
    }
}

// TODO write a macro that does this automatically for any enum
static WINDOW_TYPES: [WindowType; 5] = [
    WindowType::Hanning,
    WindowType::Hamming,
    WindowType::Blackman,
    WindowType::Nuttall,
    WindowType::None,
];

impl WindowType {
    pub fn values() -> [WindowType; 5] {
        WINDOW_TYPES
    }

    pub fn as_window_vec<T: FromF64>(self, window_size: usize) -> Option<Vec<T>> {
        match self {
            WindowType::Hanning => Some(
                apodize::hanning_iter(window_size)
                    .map(FromF64::from_f64)
                    .collect(),
            ),
            WindowType::Hamming => Some(
                apodize::hamming_iter(window_size)
                    .map(FromF64::from_f64)
                    .collect(),
            ),
            WindowType::Blackman => Some(
                apodize::blackman_iter(window_size)
                    .map(FromF64::from_f64)
                    .collect(),
            ),
            WindowType::Nuttall => Some(
                apodize::nuttall_iter(window_size)
                    .map(FromF64::from_f64)
                    .collect(),
            ),
            WindowType::None => None,
        }
    }
}

pub trait FromF64 {
    fn from_f64(n: f64) -> Self;
}
impl FromF64 for f64 {
    fn from_f64(n: f64) -> Self {
        n
    }
}
impl FromF64 for f32 {
    fn from_f64(n: f64) -> Self {
        n as f32
    }
}

/// returns `0` if `log10(value).is_negative()`.
/// otherwise returns `log10(value)`.
/// `log10` turns values in domain `0..1` into values
/// in range `-inf..0`.
/// `log10_positive` turns values in domain `0..1` into `0`.
/// this sets very small values to zero which may not be
/// what you want depending on your application.
#[inline]
pub fn log10_positive<T: Float + Signed + Zero>(value: T) -> T {
    // Float.log10
    // Signed.is_negative
    // Zero.zero
    let log = value.log10();
    if log.is_negative() {
        T::zero()
    } else {
        log
    }
}
