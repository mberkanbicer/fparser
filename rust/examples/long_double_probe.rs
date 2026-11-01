//! Long-double backend probe: read one expression per line and evaluate it at
//! `x = 0.5`, printing the `f64` projection for the native differential.
//!
//! Requires `--features long-double`.

use fparser::{real_from_f64, real_to_f64, Parser};
use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("stdin read");
    for expression in input.lines().filter(|line| !line.trim().is_empty()) {
        let mut parser = Parser::new();
        match parser.parse(expression, "x", false) {
            Ok(()) => {
                let value = parser
                    .eval(&[real_from_f64(0.5)])
                    .unwrap_or_else(|error| panic!("{expression}: {error:?}"));
                println!("{:.17e}", real_to_f64(&value));
            }
            Err(error) => println!("ERROR {error:?}"),
        }
    }
}
