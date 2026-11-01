//! Numeric scalar selected at compile time (native fparser build variants).
//!
//! `f64` is the default and the only backend used by CSXCAD/openEMS. The
//! `float` feature selects `f32` for the native `float` variant; the
//! `long-double`/`mpfr` variants land on top of the same seam.

#[cfg(feature = "float")]
pub type Real = f32;

#[cfg(not(feature = "float"))]
pub type Real = f64;

/// Convert an `f64` literal/constant to the active scalar.
#[inline]
pub fn from_f64(value: f64) -> Real {
    value as Real
}

/// Convert the active scalar to `f64` (for bit storage and reporting).
#[inline]
pub fn to_f64(value: Real) -> f64 {
    value as f64
}

/// Constants at the active precision.
pub mod consts {
    use super::Real;

    pub const LN_2: Real = std::f64::consts::LN_2 as Real;
    pub const LOG2_E: Real = std::f64::consts::LOG2_E as Real;
    pub const LOG10_E: Real = std::f64::consts::LOG10_E as Real;
    pub const LOG10_2: Real = std::f64::consts::LOG10_2 as Real;
    pub const FRAC_PI_2: Real = std::f64::consts::FRAC_PI_2 as Real;
    pub const PI: Real = std::f64::consts::PI as Real;
}
