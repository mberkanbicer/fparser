use fparser::Parser;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cases = match std::env::args_os().nth(1) {
        Some(path) => std::fs::read_to_string(path)?,
        None => include_str!("../../tests/differential_cases.tsv").to_owned(),
    };
    for (line_number, line) in cases.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 5, "invalid corpus row {}", line_number + 1);

        let mut parser = Parser::new();
        assert!(parser.add_constant("pi", std::f64::consts::PI));
        assert!(parser.add_unit("cm", 0.01));
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_from_function = Arc::clone(&calls);
        assert!(parser.add_function("count", 1, move |values| {
            calls_from_function.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        let sequence = Arc::new(AtomicUsize::new(0));
        let sequence_value = Arc::clone(&sequence);
        let sequence_calls = Arc::clone(&calls);
        assert!(parser.add_function("sequence", 1, move |_| {
            sequence_calls.fetch_add(1, Ordering::SeqCst);
            sequence_value.fetch_add(1, Ordering::SeqCst) as f64 + 1.0
        }));
        let mut square = Parser::new();
        square.parse("x^2", "x", false).unwrap();
        assert!(parser.add_parser_function("square", Arc::new(square)));
        let mut sum = Parser::new();
        sum.parse("a+b", "a,b", false).unwrap();
        assert!(parser.add_parser_function("sum2", Arc::new(sum)));
        let mut negative_root = Parser::new();
        negative_root.parse("sqrt(-x)", "x", false).unwrap();
        assert!(parser.add_parser_function("root", Arc::new(negative_root)));
        let degrees = fields[3] == "1";
        if fields[2] == "@deduce" {
            match parser.parse_and_deduce_variables(fields[1], degrees) {
                Ok(variables) => println!("{}\tvars\t{}\t-1", fields[0], variables.join(",")),
                Err(error) => println!(
                    "{}\tparse\t{}\t{}",
                    fields[0],
                    error.position,
                    error.kind.legacy_code()
                ),
            }
            continue;
        }
        let (variables, optimize) = fields[2]
            .strip_prefix("@optimize:")
            .map(|variables| (variables, true))
            .unwrap_or((fields[2], false));
        if let Err(error) = parser.parse(fields[1], variables, degrees) {
            println!(
                "{}\tparse\t{}\t{}",
                fields[0],
                error.position,
                error.kind.legacy_code()
            );
            continue;
        }
        if optimize {
            parser.optimize();
        }

        let values = if fields[4] == "-" {
            Vec::new()
        } else {
            fields[4]
                .split(',')
                .map(|value| value.parse::<f64>().expect("valid numeric test value"))
                .collect()
        };
        match parser.eval(&values) {
            Ok(value) => println!("{}\teval\t{value:.17}\t0", fields[0]),
            Err(error) => println!(
                "{}\teval\t-\t{}",
                fields[0],
                error.legacy_code().unwrap_or(0)
            ),
        }
        println!(
            "{}.calls\tcalls\t{}\t0",
            fields[0],
            calls.load(Ordering::SeqCst)
        );
    }
    Ok(())
}
