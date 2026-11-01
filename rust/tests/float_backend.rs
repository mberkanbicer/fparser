#![cfg(feature = "float")]
//! The native `float` build: the whole parser evaluates in `f32`.
//!
//! The f64 unit-test module is compiled out for this feature; these tests cover
//! the f32 path, including the precision the native float build exposes.

use fparser::{EvalError, Parser};

#[test]
fn arithmetic_is_evaluated_in_f32() {
    let mut parser = Parser::new();
    parser.parse("x^2 + 1", "x", false).unwrap();
    let value = parser.eval(&[2.0]).unwrap();
    assert_eq!(value, 5.0f32);
    assert_eq!(std::mem::size_of_val(&value), 4);
}

#[test]
fn functions_keep_f32_precision() {
    let mut parser = Parser::new();
    assert!(parser.add_constant("pi", std::f32::consts::PI));
    parser.parse("sin(pi/2)", "", false).unwrap();
    let value = parser.eval(&[]).unwrap();
    assert!((value - 1.0f32).abs() < 1.0e-6);
}

#[test]
fn domain_errors_are_reported() {
    let mut parser = Parser::new();
    parser.parse("sqrt(x)", "x", false).unwrap();
    assert_eq!(parser.eval(&[-1.0]), Err(EvalError::NegativeSquareRoot));
}
