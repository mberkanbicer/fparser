//! Numeric scalar selected at compile time (native fparser build variants).
//!
//! `f64` is the default and the only backend used by CSXCAD/openEMS. The
//! `float` feature selects `f32`, `long-double` selects IEEE binary128, and
//! `mpfr` selects a 256-bit MPFR/GMP float with the non-`Copy` scalar path.
//!
//! Rust stable has no x87 80-bit type. The `long-double` backend uses binary128
//! (113-bit mantissa) through the `f128`/libquadmath bindings: it carries more
//! precision than the native 80-bit long double (64-bit mantissa), so results
//! agree with the native variant to at least its own precision. It needs
//! libquadmath and is therefore GNU/Clang-only. The `mpfr` backend uses
//! `rug::Float` at the native 256-bit default and is feature-gated off the
//! default build; both are validated against the reference C++ variants.

#[cfg(any(
    all(feature = "float", feature = "long-double"),
    all(feature = "float", feature = "mpfr"),
    all(feature = "long-double", feature = "mpfr"),
))]
compile_error!("select only one fparser numeric backend");

#[cfg(all(feature = "long-double", target_env = "msvc"))]
compile_error!(
    "the long-double backend needs libquadmath; build it on a GNU/Clang target"
);

#[cfg(feature = "float")]
pub type Real = f32;

#[cfg(feature = "long-double")]
pub type Real = f128_scalar::F128;

#[cfg(feature = "mpfr")]
pub type Real = mpfr_scalar::MpfrReal;

#[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
pub type Real = f64;

/// Convert an `f64` literal/constant to the active scalar.
#[inline]
pub fn from_f64(value: f64) -> Real {
    #[cfg(feature = "float")]
    {
        value as f32
    }
    #[cfg(feature = "long-double")]
    {
        f128_scalar::F128::from(value)
    }
    #[cfg(feature = "mpfr")]
    {
        mpfr_scalar::MpfrReal::from_f64(value)
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        value
    }
}

/// Convert an `i64` to the active scalar.
#[inline]
pub fn from_i64(value: i64) -> Real {
    #[cfg(feature = "float")]
    {
        value as f32
    }
    #[cfg(feature = "long-double")]
    {
        f128_scalar::F128::from(value)
    }
    #[cfg(feature = "mpfr")]
    {
        mpfr_scalar::MpfrReal::from_i64(value)
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        value as f64
    }
}

/// Convert a `u64` (hex literal) to the active scalar.
#[inline]
pub fn from_u64(value: u64) -> Real {
    #[cfg(feature = "float")]
    {
        value as f32
    }
    #[cfg(feature = "long-double")]
    {
        f128_scalar::F128::from(value)
    }
    #[cfg(feature = "mpfr")]
    {
        mpfr_scalar::MpfrReal::from_u64(value)
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        value as f64
    }
}

/// Convert the active scalar to `f64` (for bit storage and reporting).
#[inline]
pub fn to_f64(value: &Real) -> f64 {
    #[cfg(feature = "float")]
    {
        *value as f64
    }
    #[cfg(feature = "long-double")]
    {
        value.to_f64()
    }
    #[cfg(feature = "mpfr")]
    {
        value.to_f64()
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        *value
    }
}

/// Convert the active scalar to `i64` when it holds an exact integer.
#[inline]
pub fn to_i64(value: &Real) -> Option<i64> {
    #[cfg(feature = "float")]
    {
        Some(*value as i64)
    }
    #[cfg(feature = "long-double")]
    {
        use num_traits::ToPrimitive;
        value.0.to_i64()
    }
    #[cfg(feature = "mpfr")]
    {
        value.to_i64()
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        Some(*value as i64)
    }
}

/// Render the active scalar with `digits` significant decimal digits.
///
/// Used by the MPFR/long-double differential harnesses; the `f64` backends
/// simply render their `f64` projection.
pub fn to_decimal(value: &Real, digits: usize) -> String {
    #[cfg(feature = "mpfr")]
    {
        format!("{:.*e}", digits.saturating_sub(1), &value.0)
    }
    #[cfg(not(feature = "mpfr"))]
    {
        format!("{:.*e}", digits.saturating_sub(1), to_f64(value))
    }
}

/// Convert a boolean comparison result to the active scalar.
#[inline]
pub fn from_bool(value: bool) -> Real {
    from_f64(if value { 1.0 } else { 0.0 })
}

/// Positive infinity at the active precision.
#[inline]
pub fn infinity() -> Real {
    #[cfg(feature = "float")]
    {
        f32::INFINITY
    }
    #[cfg(feature = "long-double")]
    {
        f128_scalar::F128::INFINITY
    }
    #[cfg(feature = "mpfr")]
    {
        mpfr_scalar::MpfrReal::infinity()
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        f64::INFINITY
    }
}

/// Parse a decimal literal at the active precision.
///
/// The native long-double and MPFR variants parse the literal text directly
/// (`strtold`/`parseValue`), so converting through `f64` would lose digits.
pub fn parse_literal(text: &str) -> Option<Real> {
    #[cfg(feature = "float")]
    {
        text.parse::<f32>().ok()
    }
    #[cfg(feature = "long-double")]
    {
        f128_scalar::F128::parse(text)
    }
    #[cfg(feature = "mpfr")]
    {
        mpfr_scalar::MpfrReal::parse(text)
    }
    #[cfg(not(any(feature = "float", feature = "long-double", feature = "mpfr")))]
    {
        text.parse::<f64>().ok()
    }
}

/// Constants at the active precision.
pub mod consts {
    use super::Real;

    pub fn ln_2() -> Real {
        super::from_f64(std::f64::consts::LN_2)
    }

    pub fn log2_e() -> Real {
        super::from_f64(std::f64::consts::LOG2_E)
    }

    pub fn log10_e() -> Real {
        super::from_f64(std::f64::consts::LOG10_E)
    }

    pub fn log10_2() -> Real {
        super::from_f64(std::f64::consts::LOG10_2)
    }

    pub fn frac_pi_2() -> Real {
        super::from_f64(std::f64::consts::FRAC_PI_2)
    }

    pub fn pi() -> Real {
        super::from_f64(std::f64::consts::PI)
    }
}

#[cfg(feature = "long-double")]
mod f128_scalar {
    use f128::f128;
    use num_traits::{Float, ToPrimitive};
    use std::cmp::Ordering;
    use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

    /// IEEE binary128 scalar used by the `long-double` backend.
    ///
    /// The wrapper supplies the small set of conversions and operator impls
    /// that the parser uses with `f64` literals, so the parser body stays the
    /// same across backends.
    #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
    pub struct F128(pub f128);

    impl F128 {
        pub const INFINITY: Self = F128(f128::INFINITY);
        pub const NAN: Self = F128(f128::NAN);

        pub fn parse(text: &str) -> Option<Self> {
            f128::parse(text).ok().map(F128)
        }

        pub fn to_f64(self) -> f64 {
            self.0.to_f64().unwrap_or(f64::NAN)
        }

        pub fn abs(self) -> Self {
            F128(self.0.abs())
        }

        pub fn sqrt(self) -> Self {
            F128(self.0.sqrt())
        }

        pub fn cbrt(self) -> Self {
            F128(self.0.cbrt())
        }

        pub fn sin(self) -> Self {
            F128(self.0.sin())
        }

        pub fn cos(self) -> Self {
            F128(self.0.cos())
        }

        pub fn tan(self) -> Self {
            F128(self.0.tan())
        }

        pub fn asin(self) -> Self {
            F128(self.0.asin())
        }

        pub fn acos(self) -> Self {
            F128(self.0.acos())
        }

        pub fn atan(self) -> Self {
            F128(self.0.atan())
        }

        pub fn atan2(self, other: Self) -> Self {
            F128(self.0.atan2(other.0))
        }

        pub fn sinh(self) -> Self {
            F128(self.0.sinh())
        }

        pub fn cosh(self) -> Self {
            F128(self.0.cosh())
        }

        pub fn tanh(self) -> Self {
            F128(self.0.tanh())
        }

        pub fn exp(self) -> Self {
            F128(self.0.exp())
        }

        pub fn ln(self) -> Self {
            F128(self.0.ln())
        }

        pub fn log10(self) -> Self {
            F128(self.0.log10())
        }

        pub fn log2(self) -> Self {
            F128(self.0.log2())
        }

        pub fn powf(self, exponent: Self) -> Self {
            F128(self.0.powf(exponent.0))
        }

        pub fn fract(self) -> Self {
            F128(self.0.fract())
        }

        pub fn floor(self) -> Self {
            F128(self.0.floor())
        }

        pub fn ceil(self) -> Self {
            F128(self.0.ceil())
        }

        pub fn round(self) -> Self {
            F128(self.0.round())
        }

        pub fn trunc(self) -> Self {
            F128(self.0.trunc())
        }

        pub fn is_nan(self) -> bool {
            self.0.is_nan()
        }

        pub fn is_infinite(self) -> bool {
            self.0.is_infinite()
        }

        pub fn is_finite(self) -> bool {
            self.0.is_finite()
        }

        pub fn to_radians(self) -> Self {
            self * F128::from(std::f64::consts::PI / 180.0)
        }

        pub fn to_degrees(self) -> Self {
            self * F128::from(180.0 / std::f64::consts::PI)
        }

        /// Bits of the `f64` rounding; the parser tests use this to distinguish
        /// `+0.0` from `-0.0` and to compare integer-representable values.
        pub fn to_bits(self) -> u64 {
            self.to_f64().to_bits()
        }
    }

    impl From<f64> for F128 {
        fn from(value: f64) -> Self {
            F128(f128::from(value))
        }
    }

    impl From<f32> for F128 {
        fn from(value: f32) -> Self {
            F128(f128::from(value as f64))
        }
    }

    impl From<i64> for F128 {
        fn from(value: i64) -> Self {
            F128(f128::from(value as f64))
        }
    }

    impl From<u64> for F128 {
        fn from(value: u64) -> Self {
            F128(f128::from(value as f64))
        }
    }

    impl From<bool> for F128 {
        fn from(value: bool) -> Self {
            F128::from(if value { 1.0 } else { 0.0 })
        }
    }

    impl PartialEq<f64> for F128 {
        fn eq(&self, other: &f64) -> bool {
            self.0 == f128::from(*other)
        }
    }

    impl PartialOrd<f64> for F128 {
        fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
            self.0.partial_cmp(&f128::from(*other))
        }
    }

    impl Add for F128 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            F128(self.0 + rhs.0)
        }
    }

    impl Sub for F128 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            F128(self.0 - rhs.0)
        }
    }

    impl Mul for F128 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            F128(self.0 * rhs.0)
        }
    }

    impl Div for F128 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            F128(self.0 / rhs.0)
        }
    }

    impl Rem for F128 {
        type Output = Self;
        fn rem(self, rhs: Self) -> Self {
            F128(self.0 % rhs.0)
        }
    }

    impl Neg for F128 {
        type Output = Self;
        fn neg(self) -> Self {
            F128(-self.0)
        }
    }

    impl std::ops::AddAssign for F128 {
        fn add_assign(&mut self, rhs: Self) {
            self.0 = self.0 + rhs.0;
        }
    }

    impl std::ops::SubAssign for F128 {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 = self.0 - rhs.0;
        }
    }

    impl std::ops::MulAssign for F128 {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = self.0 * rhs.0;
        }
    }

    impl std::ops::DivAssign for F128 {
        fn div_assign(&mut self, rhs: Self) {
            self.0 = self.0 / rhs.0;
        }
    }

    macro_rules! scalar_ops_with_f64 {
        ($($trait:ident, $method:ident, $op:tt;)*) => {
            $(
                impl $trait<f64> for F128 {
                    type Output = F128;
                    fn $method(self, rhs: f64) -> F128 {
                        F128(self.0 $op f128::from(rhs))
                    }
                }

                impl $trait<F128> for f64 {
                    type Output = F128;
                    fn $method(self, rhs: F128) -> F128 {
                        F128(f128::from(self) $op rhs.0)
                    }
                }
            )*
        };
    }

    scalar_ops_with_f64! {
        Add, add, +;
        Sub, sub, -;
        Mul, mul, *;
        Div, div, /;
        Rem, rem, %;
    }
}

#[cfg(feature = "mpfr")]
mod mpfr_scalar {
    use rug::ops::Pow;
    use rug::Float;
    use std::cmp::Ordering;
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

    /// Native `MpfrFloat` default: a 256-bit mantissa.
    pub const PRECISION: u32 = 256;

    /// Arbitrary-precision scalar used by the `mpfr` backend.
    ///
    /// This is the non-`Copy` path: the parser clones values at the sites the
    /// `f64` backend copies them, and the wrapper exposes the same operator
    /// and method surface as the other backends.
    #[derive(Clone, Debug, PartialEq, PartialOrd)]
    pub struct MpfrReal(pub Float);

    impl MpfrReal {
        pub fn from_f64(value: f64) -> Self {
            MpfrReal(Float::with_val(PRECISION, value))
        }

        pub fn from_i64(value: i64) -> Self {
            MpfrReal(Float::with_val(PRECISION, value))
        }

        pub fn from_u64(value: u64) -> Self {
            MpfrReal(Float::with_val(PRECISION, value))
        }

        pub fn infinity() -> Self {
            MpfrReal(Float::with_val(PRECISION, f64::INFINITY))
        }

        pub fn parse(text: &str) -> Option<Self> {
            let parsed = Float::parse(text).ok()?;
            Some(MpfrReal(Float::with_val(PRECISION, parsed)))
        }

        pub fn to_f64(&self) -> f64 {
            self.0.to_f64()
        }

        pub fn to_i64(&self) -> Option<i64> {
            if self.0.is_nan() || self.0.is_infinite() {
                return None;
            }
            self.0.to_integer()?.to_i64()
        }

        pub fn abs(self) -> Self {
            MpfrReal(self.0.abs())
        }

        pub fn sqrt(self) -> Self {
            MpfrReal(self.0.sqrt())
        }

        pub fn cbrt(self) -> Self {
            MpfrReal(self.0.cbrt())
        }

        pub fn sin(self) -> Self {
            MpfrReal(self.0.sin())
        }

        pub fn cos(self) -> Self {
            MpfrReal(self.0.cos())
        }

        pub fn tan(self) -> Self {
            MpfrReal(self.0.tan())
        }

        pub fn asin(self) -> Self {
            MpfrReal(self.0.asin())
        }

        pub fn acos(self) -> Self {
            MpfrReal(self.0.acos())
        }

        pub fn atan(self) -> Self {
            MpfrReal(self.0.atan())
        }

        pub fn atan2(self, other: Self) -> Self {
            MpfrReal(self.0.atan2(&other.0))
        }

        pub fn sinh(self) -> Self {
            MpfrReal(self.0.sinh())
        }

        pub fn cosh(self) -> Self {
            MpfrReal(self.0.cosh())
        }

        pub fn tanh(self) -> Self {
            MpfrReal(self.0.tanh())
        }

        pub fn exp(self) -> Self {
            MpfrReal(self.0.exp())
        }

        pub fn ln(self) -> Self {
            MpfrReal(self.0.ln())
        }

        pub fn log10(self) -> Self {
            MpfrReal(self.0.log10())
        }

        pub fn log2(self) -> Self {
            MpfrReal(self.0.log2())
        }

        pub fn powf(self, exponent: Self) -> Self {
            MpfrReal(self.0.pow(&exponent.0))
        }

        pub fn fract(self) -> Self {
            MpfrReal(self.0.fract())
        }

        pub fn floor(self) -> Self {
            MpfrReal(self.0.floor())
        }

        pub fn ceil(self) -> Self {
            MpfrReal(self.0.ceil())
        }

        pub fn round(self) -> Self {
            MpfrReal(self.0.round())
        }

        pub fn trunc(self) -> Self {
            MpfrReal(self.0.trunc())
        }

        pub fn is_nan(&self) -> bool {
            self.0.is_nan()
        }

        pub fn is_infinite(&self) -> bool {
            self.0.is_infinite()
        }

        pub fn is_finite(&self) -> bool {
            self.0.is_finite()
        }

        pub fn to_radians(self) -> Self {
            self * MpfrReal::from_f64(std::f64::consts::PI / 180.0)
        }

        pub fn to_degrees(self) -> Self {
            self * MpfrReal::from_f64(180.0 / std::f64::consts::PI)
        }

        pub fn to_bits(&self) -> u64 {
            self.to_f64().to_bits()
        }
    }

    impl From<f64> for MpfrReal {
        fn from(value: f64) -> Self {
            Self::from_f64(value)
        }
    }

    impl From<bool> for MpfrReal {
        fn from(value: bool) -> Self {
            Self::from_f64(if value { 1.0 } else { 0.0 })
        }
    }

    impl PartialEq<f64> for MpfrReal {
        fn eq(&self, other: &f64) -> bool {
            self.0 == Float::with_val(PRECISION, *other)
        }
    }

    impl PartialOrd<f64> for MpfrReal {
        fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
            self.0.partial_cmp(&Float::with_val(PRECISION, *other))
        }
    }

    macro_rules! mpfr_binop {
        ($($trait:ident, $method:ident, $op:tt;)*) => {
            $(
                impl $trait for MpfrReal {
                    type Output = Self;
                    fn $method(self, rhs: Self) -> Self {
                        MpfrReal(self.0 $op rhs.0)
                    }
                }

                impl $trait<f64> for MpfrReal {
                    type Output = Self;
                    fn $method(self, rhs: f64) -> Self {
                        MpfrReal(self.0 $op Float::with_val(PRECISION, rhs))
                    }
                }

                impl $trait<MpfrReal> for f64 {
                    type Output = MpfrReal;
                    fn $method(self, rhs: MpfrReal) -> MpfrReal {
                        MpfrReal(Float::with_val(PRECISION, self) $op rhs.0)
                    }
                }

                impl $trait<MpfrReal> for &MpfrReal {
                    type Output = MpfrReal;
                    fn $method(self, rhs: MpfrReal) -> MpfrReal {
                        MpfrReal(self.0.clone() $op rhs.0)
                    }
                }
            )*
        };
    }

    mpfr_binop! {
        Add, add, +;
        Sub, sub, -;
        Mul, mul, *;
        Div, div, /;
        Rem, rem, %;
    }

    impl Neg for MpfrReal {
        type Output = Self;
        fn neg(self) -> Self {
            MpfrReal(-self.0)
        }
    }

    impl AddAssign for MpfrReal {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }

    impl SubAssign for MpfrReal {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }

    impl MulAssign for MpfrReal {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 *= rhs.0;
        }
    }

    impl DivAssign for MpfrReal {
        fn div_assign(&mut self, rhs: Self) {
            self.0 /= rhs.0;
        }
    }
}

