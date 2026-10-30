use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const DEFAULT_EPSILON: f64 = 1.0e-12;
static EPSILON_BITS: AtomicU64 = AtomicU64::new(DEFAULT_EPSILON.to_bits());

type UserFunction = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;

#[derive(Clone)]
enum FunctionTarget {
    Callback(UserFunction),
    Parser(Arc<Parser>),
}

#[derive(Clone)]
struct FunctionDefinition {
    parameters: usize,
    target: FunctionTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseErrorKind {
    Syntax,
    MismatchedParenthesis,
    MissingParenthesis,
    EmptyParentheses,
    ExpectedOperator,
    OutOfMemory,
    UnexpectedError,
    InvalidVariables,
    InvalidFunctionParameters,
    PrematureEnd,
    ExpectedFunctionParenthesis,
    UnknownIdentifier,
    NoFunctionParsedYet,
    FpNoError,
}

impl ParseErrorKind {
    pub fn legacy_code(self) -> u8 {
        match self {
            Self::Syntax => 0,
            Self::MismatchedParenthesis => 1,
            Self::MissingParenthesis => 2,
            Self::EmptyParentheses => 3,
            Self::ExpectedOperator => 4,
            Self::OutOfMemory => 5,
            Self::UnexpectedError => 6,
            Self::InvalidVariables => 7,
            Self::InvalidFunctionParameters => 8,
            Self::PrematureEnd => 9,
            Self::ExpectedFunctionParenthesis => 10,
            Self::UnknownIdentifier => 11,
            Self::NoFunctionParsedYet => 12,
            Self::FpNoError => 13,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub position: usize,
}

const ParseErrorMessage: &[&str; 14] = &[
    "Syntax error",
    "Mismatched parenthesis",
    "Missing ')'",
    "Empty parentheses",
    "Syntax error: Operator expected",
    "Not enough memory",
    "An unexpected error occurred. Please make a full bug report to the author",
    "Syntax error in parameter 'Vars' given to FunctionParser::Parse()",
    "Illegal number of parameters to function",
    "Syntax error: Premature end of string",
    "Syntax error: Expecting ( after function",
    "Unknown identifier",
    "(No function has been parsed yet)",
    "",
];

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = ParseErrorMessage[self.kind.legacy_code() as usize];
        write!(formatter, "{0} at byte {1}", msg, self.position)
    }
}

impl std::error::Error for ParseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalError {
    DivisionByZero,
    NegativeSquareRoot,
    InvalidLogarithmOrPower,
    InvalidInverseTrigonometricInput,
    NotParsed,
    InvalidVariableCount { expected: usize, actual: usize },
}

impl EvalError {
    pub fn legacy_code(self) -> Option<u8> {
        match self {
            Self::DivisionByZero => Some(1),
            Self::NegativeSquareRoot => Some(2),
            Self::InvalidLogarithmOrPower => Some(3),
            Self::InvalidInverseTrigonometricInput => Some(4),
            Self::NotParsed | Self::InvalidVariableCount { .. } => None,
        }
    }
}

impl fmt::Display for EvalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => formatter.write_str("division by zero"),
            Self::NegativeSquareRoot => formatter.write_str("square root of a negative value"),
            Self::InvalidLogarithmOrPower => {
                formatter.write_str("invalid logarithm or zero-to-negative power")
            }
            Self::InvalidInverseTrigonometricInput => {
                formatter.write_str("invalid inverse trigonometric input")
            }
            Self::NotParsed => formatter.write_str("no expression has been parsed"),
            Self::InvalidVariableCount { expected, actual } => {
                write!(
                    formatter,
                    "expected {expected} variables, received {actual}"
                )
            }
        }
    }
}

impl std::error::Error for EvalError {}

#[derive(Clone)]
enum Expression {
    Number(f64),
    Variable(usize),
    Inline(usize),
    Shared(usize, Box<Expression>),
    Unary(UnaryOperator, Box<Expression>),
    Binary(BinaryOperator, Box<Expression>, Box<Expression>),
    Builtin(Builtin, Vec<Expression>),
    UserFunction(FunctionTarget, Vec<Expression>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnaryOperator {
    Negate,
    Not,
    ExactZero,
    ExactNegative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Builtin {
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Cbrt,
    Ceil,
    Cos,
    Cosh,
    Cot,
    Csc,
    Exp,
    Exp2,
    Floor,
    Hypot,
    If,
    Int,
    Log,
    Log10,
    Log2,
    Max,
    Min,
    Pow,
    Sec,
    Sin,
    Sinh,
    Sqrt,
    Tan,
    Tanh,
    Trunc,
}

impl Builtin {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "abs" => Self::Abs,
            "acos" => Self::Acos,
            "acosh" => Self::Acosh,
            "asin" => Self::Asin,
            "asinh" => Self::Asinh,
            "atan" => Self::Atan,
            "atan2" => Self::Atan2,
            "atanh" => Self::Atanh,
            "cbrt" => Self::Cbrt,
            "ceil" => Self::Ceil,
            "cos" => Self::Cos,
            "cosh" => Self::Cosh,
            "cot" => Self::Cot,
            "csc" => Self::Csc,
            "exp" => Self::Exp,
            "exp2" => Self::Exp2,
            "floor" => Self::Floor,
            "hypot" => Self::Hypot,
            "if" => Self::If,
            "int" => Self::Int,
            "log" => Self::Log,
            "log10" => Self::Log10,
            "log2" => Self::Log2,
            "max" => Self::Max,
            "min" => Self::Min,
            "pow" => Self::Pow,
            "sec" => Self::Sec,
            "sin" => Self::Sin,
            "sinh" => Self::Sinh,
            "sqrt" => Self::Sqrt,
            "tan" => Self::Tan,
            "tanh" => Self::Tanh,
            "trunc" => Self::Trunc,
            _ => return None,
        })
    }

    fn parameter_count(self) -> usize {
        match self {
            Self::Atan2 | Self::Hypot | Self::Max | Self::Min | Self::Pow => 2,
            Self::If => 3,
            _ => 1,
        }
    }

    fn angle_input(self) -> bool {
        matches!(
            self,
            Self::Cos
                | Self::Cosh
                | Self::Cot
                | Self::Csc
                | Self::Sec
                | Self::Sin
                | Self::Sinh
                | Self::Tan
                | Self::Tanh
        )
    }

    fn angle_output(self) -> bool {
        matches!(
            self,
            Self::Acos | Self::Acosh | Self::Asin | Self::Asinh | Self::Atan | Self::Atan2
        )
    }
}

#[derive(Clone)]
struct Program {
    variables: Vec<String>,
    inline_bindings: Vec<Expression>,
    expression: Expression,
    use_degrees: bool,
}

#[derive(Clone)]
pub struct Parser {
    constants: HashMap<String, f64>,
    units: HashMap<String, f64>,
    functions: HashMap<String, FunctionDefinition>,
    program: Option<Program>,
    delimiter: Option<char>,
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Parser {
    pub fn new() -> Self {
        Self {
            constants: HashMap::new(),
            units: HashMap::new(),
            functions: HashMap::new(),
            program: None,
            delimiter: None,
        }
    }

    fn identifier_exists(&self, name: &str) -> bool {
        self.constants.contains_key(name)
            || self.units.contains_key(name)
            || self.functions.contains_key(name)
    }

    pub fn add_constant(&mut self, name: &str, value: f64) -> bool {
        if !is_identifier(name) || is_reserved_name(name) || self.identifier_exists(name) {
            return false;
        }
        self.constants.insert(name.to_owned(), value);
        true
    }

    pub fn add_unit(&mut self, name: &str, value: f64) -> bool {
        if !is_identifier(name) || is_reserved_name(name) || self.identifier_exists(name) {
            return false;
        }
        self.units.insert(name.to_owned(), value);
        true
    }

    pub fn add_function<F>(&mut self, name: &str, parameters: usize, function: F) -> bool
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        if !is_identifier(name) || is_reserved_name(name) || self.identifier_exists(name) {
            return false;
        }
        self.functions.insert(
            name.to_owned(),
            FunctionDefinition {
                parameters,
                target: FunctionTarget::Callback(Arc::new(function)),
            },
        );
        true
    }

    /// Links a compiled child parser; its variable order defines argument order.
    /// Returns false for an invalid or colliding name, or an unparsed child.
    pub fn add_parser_function(&mut self, name: &str, parser: Arc<Parser>) -> bool {
        if !is_identifier(name) || is_reserved_name(name) || self.identifier_exists(name) {
            return false;
        }
        let Some(program) = parser.program.as_ref() else {
            return false;
        };
        self.functions.insert(
            name.to_owned(),
            FunctionDefinition {
                parameters: program.variables.len(),
                target: FunctionTarget::Parser(parser),
            },
        );
        true
    }

    pub fn remove_identifier(&mut self, name: &str) -> bool {
        self.constants.remove(name).is_some()
            || self.units.remove(name).is_some()
            || self.functions.remove(name).is_some()
    }

    pub fn set_epsilon(epsilon: f64) {
        EPSILON_BITS.store(epsilon.to_bits(), Ordering::Relaxed);
    }

    pub fn epsilon() -> f64 {
        f64::from_bits(EPSILON_BITS.load(Ordering::Relaxed))
    }

    pub fn set_delimiter_char(&mut self, delimiter: Option<char>) {
        self.delimiter = delimiter;
    }

    pub fn parse(
        &mut self,
        expression: &str,
        variables: &str,
        use_degrees: bool,
    ) -> Result<(), ParseError> {
        self.program = None;
        let variable_names =
            parse_variable_list(variables, &self.constants, &self.units, &self.functions).map_err(
                |kind| ParseError {
                    kind,
                    position: expression.len(),
                },
            )?;
        let mut grammar = Grammar::new(
            expression,
            variable_names,
            &self.constants,
            &self.units,
            &self.functions,
        )?;
        let (inline_bindings, parsed, consumed) = grammar.parse_program()?;
        if consumed < expression.len() {
            let next = expression[consumed..].chars().next();
            if self.delimiter.is_none() || next != self.delimiter {
                return Err(ParseError {
                    kind: ParseErrorKind::ExpectedOperator,
                    position: consumed,
                });
            }
        }
        self.program = Some(Program {
            variables: grammar.variables,
            inline_bindings: inline_bindings
                .into_iter()
                .map(rewrite_compile_patterns)
                .collect(),
            expression: rewrite_compile_patterns(parsed),
            use_degrees,
        });
        Ok(())
    }

    pub fn parse_and_deduce_variables(
        &mut self,
        expression: &str,
        use_degrees: bool,
    ) -> Result<Vec<String>, ParseError> {
        let tokens = tokenize(expression)?;
        let inline_names = tokens
            .windows(2)
            .filter_map(|pair| match (&pair[0].kind, &pair[1].kind) {
                (TokenKind::Identifier(name), TokenKind::Assign) => Some(name.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut variables = Vec::new();
        for token in &tokens {
            let TokenKind::Identifier(name) = &token.kind else {
                continue;
            };
            if inline_names.contains(name) {
                continue;
            }
            if Builtin::from_name(name).is_some()
                || self.constants.contains_key(name)
                || self.units.contains_key(name)
                || self.functions.contains_key(name)
                || variables.contains(name)
            {
                continue;
            }
            variables.push(name.clone());
        }
        variables.sort();
        let variable_list = variables.join(",");
        self.parse(expression, &variable_list, use_degrees)?;
        Ok(variables)
    }

    pub fn eval(&self, values: &[f64]) -> Result<f64, EvalError> {
        let program = self.program.as_ref().ok_or(EvalError::NotParsed)?;
        if values.len() != program.variables.len() {
            return Err(EvalError::InvalidVariableCount {
                expected: program.variables.len(),
                actual: values.len(),
            });
        }
        let mut inline_values = Vec::with_capacity(program.inline_bindings.len());
        let mut shared_values = HashMap::new();
        for binding in &program.inline_bindings {
            inline_values.push(evaluate(
                binding,
                values,
                &inline_values,
                program.use_degrees,
                Self::epsilon(),
                &mut shared_values,
            )?);
        }
        evaluate(
            &program.expression,
            values,
            &inline_values,
            program.use_degrees,
            Self::epsilon(),
            &mut shared_values,
        )
    }

    pub fn optimize(&mut self) {
        let Some(program) = self.program.as_mut() else {
            return;
        };
        for binding in &mut program.inline_bindings {
            *binding = fold_constants(binding.clone(), program.use_degrees, Self::epsilon());
        }
        let expression = program.expression.clone();
        if let Some(rewritten) = rewrite_nested_optimizer_expression(&expression) {
            program.expression = rewritten;
        } else {
            program.expression = fold_constants(expression, program.use_degrees, Self::epsilon());
        }
        for binding in &mut program.inline_bindings {
            *binding = rewrite_legacy_math_lowering(binding.clone());
        }
        program.expression = rewrite_legacy_math_lowering(program.expression.clone());
        let mut function_calls = Vec::new();
        for binding in &program.inline_bindings {
            collect_function_calls(binding, &mut function_calls);
        }
        collect_function_calls(&program.expression, &mut function_calls);
        for binding in &mut program.inline_bindings {
            *binding = share_repeated_functions(binding.clone(), &function_calls);
        }
        program.expression = share_repeated_functions(program.expression.clone(), &function_calls);
    }
}

fn parse_variable_list(
    input: &str,
    constants: &HashMap<String, f64>,
    units: &HashMap<String, f64>,
    functions: &HashMap<String, FunctionDefinition>,
) -> Result<Vec<String>, ParseErrorKind> {
    if input.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut variables = Vec::new();
    for name in input.split(',').map(str::trim) {
        if !is_identifier(name)
            || is_reserved_name(name)
            || constants.contains_key(name)
            || units.contains_key(name)
            || functions.contains_key(name)
            || variables.iter().any(|existing| existing == name)
        {
            return Err(ParseErrorKind::InvalidVariables);
        }
        variables.push(name.to_owned());
    }
    Ok(variables)
}

fn is_reserved_name(name: &str) -> bool {
    Builtin::from_name(name).is_some()
}

fn is_identifier(name: &str) -> bool {
    let mut characters = name.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    is_identifier_character(first, true)
        && characters.all(|character| is_identifier_character(character, false))
}

fn is_legacy_space(character: char) -> bool {
    matches!(
        character,
        '\t' | '\n' | '\u{000b}' | '\r' | ' ' | '\u{00a0}' | '\u{2000}'
            ..='\u{200b}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

fn is_identifier_character(character: char, first: bool) -> bool {
    if character.is_ascii() {
        character == '_'
            || character.is_ascii_alphabetic()
            || (!first && character.is_ascii_digit())
    } else {
        !is_legacy_space(character)
    }
}

#[derive(Clone, Debug)]
struct Token {
    kind: TokenKind,
    position: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum TokenKind {
    Number(f64),
    Identifier(String),
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    Power,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Not,
    And,
    Or,
    LeftParenthesis,
    RightParenthesis,
    Comma,
    Assign,
    Semicolon,
    Other(char),
    End,
}

fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut position = 0;
    while position < input.len() {
        let character = input[position..]
            .chars()
            .next()
            .expect("valid UTF-8 boundary");
        if is_legacy_space(character) {
            position += character.len_utf8();
            continue;
        }
        let start = position;
        if character.is_ascii_digit() || character == '.' {
            let (number, end) = parse_number(input, position).ok_or(ParseError {
                kind: ParseErrorKind::Syntax,
                position,
            })?;
            tokens.push(Token {
                kind: TokenKind::Number(number),
                position,
            });
            position = end;
            continue;
        }
        if is_identifier_character(character, true) {
            position += character.len_utf8();
            while position < input.len() {
                let next = input[position..]
                    .chars()
                    .next()
                    .expect("valid UTF-8 boundary");
                if is_identifier_character(next, false) {
                    position += next.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(Token {
                kind: TokenKind::Identifier(input[start..position].to_owned()),
                position: start,
            });
            continue;
        }
        let remaining = &input[position..];
        let (kind, length) = if remaining.starts_with(":=") {
            (TokenKind::Assign, 2)
        } else if remaining.starts_with("!=") {
            (TokenKind::NotEqual, 2)
        } else if remaining.starts_with("<=") {
            (TokenKind::LessOrEqual, 2)
        } else if remaining.starts_with(">=") {
            (TokenKind::GreaterOrEqual, 2)
        } else {
            (
                match character {
                    '+' => TokenKind::Plus,
                    '-' => TokenKind::Minus,
                    '*' => TokenKind::Multiply,
                    '/' => TokenKind::Divide,
                    '%' => TokenKind::Modulo,
                    '^' => TokenKind::Power,
                    '=' => TokenKind::Equal,
                    '!' => TokenKind::Not,
                    '<' => TokenKind::Less,
                    '>' => TokenKind::Greater,
                    '&' => TokenKind::And,
                    '|' => TokenKind::Or,
                    '(' => TokenKind::LeftParenthesis,
                    ')' => TokenKind::RightParenthesis,
                    ',' => TokenKind::Comma,
                    ';' => TokenKind::Semicolon,
                    other => TokenKind::Other(other),
                },
                character.len_utf8(),
            )
        };
        tokens.push(Token {
            kind,
            position: start,
        });
        position += length;
    }
    tokens.push(Token {
        kind: TokenKind::End,
        position: input.len(),
    });
    Ok(tokens)
}

fn parse_number(input: &str, start: usize) -> Option<(f64, usize)> {
    let bytes = input.as_bytes();
    if bytes.get(start..start + 2) == Some(b"0x") || bytes.get(start..start + 2) == Some(b"0X") {
        let mut end = start + 2;
        while bytes.get(end).is_some_and(u8::is_ascii_hexdigit) {
            end += 1;
        }
        if end == start + 2 {
            return None;
        }
        let value = u64::from_str_radix(&input[start + 2..end], 16).ok()? as f64;
        return Some((value, end));
    }

    let mut end = start;
    let mut has_digit = false;
    while bytes.get(end).is_some_and(u8::is_ascii_digit) {
        has_digit = true;
        end += 1;
    }
    if bytes.get(end) == Some(&b'.') {
        end += 1;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            has_digit = true;
            end += 1;
        }
    }
    if !has_digit {
        return None;
    }
    if matches!(bytes.get(end), Some(b'e' | b'E')) {
        let exponent_start = end;
        end += 1;
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let digit_start = end;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        if end == digit_start {
            end = exponent_start;
        }
    }
    Some((input[start..end].parse().ok()?, end))
}

/// Maximum recursion depth of the recursive-descent grammar.
///
/// A hostile expression with thousands of nested parentheses would otherwise
/// exhaust the native stack; the limit turns it into a normal parse error.
const MAX_EXPRESSION_DEPTH: usize = 512;

struct Grammar<'a> {
    tokens: Vec<Token>,
    cursor: usize,
    depth: usize,
    variables: Vec<String>,
    variable_indices: HashMap<String, usize>,
    inline_indices: HashMap<String, usize>,
    constants: &'a HashMap<String, f64>,
    units: &'a HashMap<String, f64>,
    functions: &'a HashMap<String, FunctionDefinition>,
}

impl<'a> Grammar<'a> {
    fn new(
        input: &str,
        variables: Vec<String>,
        constants: &'a HashMap<String, f64>,
        units: &'a HashMap<String, f64>,
        functions: &'a HashMap<String, FunctionDefinition>,
    ) -> Result<Self, ParseError> {
        let variable_indices = variables
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), index))
            .collect();
        Ok(Self {
            tokens: tokenize(input)?,
            cursor: 0,
            depth: 0,
            variables,
            variable_indices,
            inline_indices: HashMap::new(),
            constants,
            units,
            functions,
        })
    }

    fn parse_program(&mut self) -> Result<(Vec<Expression>, Expression, usize), ParseError> {
        let mut inline_bindings = Vec::new();
        loop {
            let assignment = matches!(
                (&self.current().kind, self.peek_kind(1)),
                (TokenKind::Identifier(_), TokenKind::Assign)
            );
            if !assignment {
                break;
            }
            let TokenKind::Identifier(name) = self.current().kind.clone() else {
                unreachable!();
            };
            if self.variable_indices.contains_key(&name)
                || self.constants.contains_key(&name)
                || self.units.contains_key(&name)
                || self.functions.contains_key(&name)
                || is_reserved_name(&name)
            {
                break;
            }
            self.advance();
            self.advance();
            let binding = self.parse_expression()?;
            self.expect(
                TokenKind::Semicolon,
                ParseErrorKind::Syntax,
                ParseErrorKind::PrematureEnd,
            )?;
            let index = inline_bindings.len();
            inline_bindings.push(binding);
            self.inline_indices.insert(name, index);
        }
        let expression = self.parse_expression()?;
        let consumed = self.current().position;
        if !matches!(self.current().kind, TokenKind::End) {
            return Ok((inline_bindings, expression, consumed));
        }
        Ok((inline_bindings, expression, consumed))
    }

    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.depth += 1;
        if self.depth > MAX_EXPRESSION_DEPTH {
            self.depth -= 1;
            return Err(ParseError {
                kind: ParseErrorKind::OutOfMemory,
                position: self.cursor,
            });
        }
        let result = self.parse_or();
        self.depth -= 1;
        result
    }

    fn parse_or(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_and()?;
        while self.consume(&TokenKind::Or) {
            let right = self.parse_and()?;
            left = Expression::Binary(BinaryOperator::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_comparison()?;
        while self.consume(&TokenKind::And) {
            let right = self.parse_comparison()?;
            left = Expression::Binary(BinaryOperator::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_comparison(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_addition()?;
        loop {
            let operator = match self.current().kind {
                TokenKind::Equal => BinaryOperator::Equal,
                TokenKind::NotEqual => BinaryOperator::NotEqual,
                TokenKind::Less => BinaryOperator::Less,
                TokenKind::LessOrEqual => BinaryOperator::LessOrEqual,
                TokenKind::Greater => BinaryOperator::Greater,
                TokenKind::GreaterOrEqual => BinaryOperator::GreaterOrEqual,
                _ => break,
            };
            self.advance();
            let right = self.parse_addition()?;
            left = Expression::Binary(operator, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_addition(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_multiplication()?;
        loop {
            let operator = match self.current().kind {
                TokenKind::Plus => BinaryOperator::Add,
                TokenKind::Minus => BinaryOperator::Subtract,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplication()?;
            left = Expression::Binary(operator, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_multiplication(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            let operator = match self.current().kind {
                TokenKind::Multiply => BinaryOperator::Multiply,
                TokenKind::Divide => BinaryOperator::Divide,
                TokenKind::Modulo => BinaryOperator::Modulo,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expression::Binary(operator, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, ParseError> {
        self.depth += 1;
        if self.depth > MAX_EXPRESSION_DEPTH {
            self.depth -= 1;
            return Err(ParseError {
                kind: ParseErrorKind::OutOfMemory,
                position: self.cursor,
            });
        }
        let result = match self.current().kind {
            TokenKind::Minus => {
                self.advance();
                Ok(Expression::Unary(
                    UnaryOperator::Negate,
                    Box::new(self.parse_unary()?),
                ))
            }
            TokenKind::Not => {
                self.advance();
                Ok(Expression::Unary(
                    UnaryOperator::Not,
                    Box::new(self.parse_unary()?),
                ))
            }
            _ => self.parse_power(),
        };
        self.depth -= 1;
        result
    }

    fn parse_power(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_primary()?;
        if let TokenKind::Identifier(name) = &self.current().kind {
            if let Some(value) = self.units.get(name) {
                let unit = *value;
                self.advance();
                left = Expression::Binary(
                    BinaryOperator::Multiply,
                    Box::new(left),
                    Box::new(Expression::Number(unit)),
                );
            }
        }
        if self.consume(&TokenKind::Power) {
            let right = self.parse_unary()?;
            left = Expression::Binary(BinaryOperator::Power, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        let token = self.current().clone();
        match token.kind {
            TokenKind::Number(value) => {
                self.advance();
                Ok(Expression::Number(value))
            }
            TokenKind::Identifier(name) => {
                self.advance();
                let builtin = Builtin::from_name(&name);
                let user_function = self.functions.get(&name).cloned();
                if builtin.is_some() || user_function.is_some() {
                    if !matches!(self.current().kind, TokenKind::LeftParenthesis) {
                        return Err(ParseError {
                            kind: ParseErrorKind::ExpectedFunctionParenthesis,
                            position: self.current().position,
                        });
                    }
                    let parameter_count = builtin
                        .map(Builtin::parameter_count)
                        .or_else(|| {
                            user_function
                                .as_ref()
                                .map(|definition| definition.parameters)
                        })
                        .expect("function definition was checked above");
                    let arguments = self.parse_arguments(parameter_count)?;
                    if let Some(builtin) = builtin {
                        Ok(Expression::Builtin(builtin, arguments))
                    } else {
                        let definition = user_function.expect("checked above");
                        Ok(Expression::UserFunction(definition.target, arguments))
                    }
                } else if let Some(index) = self.variable_indices.get(&name) {
                    Ok(Expression::Variable(*index))
                } else if let Some(index) = self.inline_indices.get(&name) {
                    Ok(Expression::Inline(*index))
                } else if let Some(value) = self.constants.get(&name) {
                    Ok(Expression::Number(*value))
                } else if self.units.contains_key(&name) {
                    Err(ParseError {
                        kind: ParseErrorKind::Syntax,
                        position: token.position,
                    })
                } else {
                    Err(ParseError {
                        kind: ParseErrorKind::UnknownIdentifier,
                        position: token.position,
                    })
                }
            }
            TokenKind::LeftParenthesis => {
                self.advance();
                if matches!(self.current().kind, TokenKind::RightParenthesis) {
                    return Err(ParseError {
                        kind: ParseErrorKind::EmptyParentheses,
                        position: self.current().position,
                    });
                }
                let expression = self.parse_expression()?;
                self.expect(
                    TokenKind::RightParenthesis,
                    ParseErrorKind::MissingParenthesis,
                    ParseErrorKind::MissingParenthesis,
                )?;
                Ok(expression)
            }
            TokenKind::RightParenthesis => Err(ParseError {
                kind: ParseErrorKind::MismatchedParenthesis,
                position: token.position,
            }),
            TokenKind::End => Err(ParseError {
                kind: ParseErrorKind::Syntax,
                position: token.position,
            }),
            _ => Err(ParseError {
                kind: ParseErrorKind::Syntax,
                position: token.position,
            }),
        }
    }

    fn parse_arguments(&mut self, parameter_count: usize) -> Result<Vec<Expression>, ParseError> {
        self.advance();
        let mut arguments = Vec::new();
        if matches!(self.current().kind, TokenKind::RightParenthesis) {
            let position = self.current().position;
            self.advance();
            if parameter_count != 0 {
                return Err(ParseError {
                    kind: ParseErrorKind::InvalidFunctionParameters,
                    position,
                });
            }
            return Ok(arguments);
        }
        loop {
            arguments.push(self.parse_expression()?);
            if matches!(self.current().kind, TokenKind::Comma) {
                let position = self.current().position;
                if arguments.len() >= parameter_count {
                    return Err(ParseError {
                        kind: ParseErrorKind::InvalidFunctionParameters,
                        position,
                    });
                }
                self.advance();
                continue;
            }
            let closing_position = self.current().position;
            self.expect(
                TokenKind::RightParenthesis,
                ParseErrorKind::MissingParenthesis,
                ParseErrorKind::MissingParenthesis,
            )?;
            if arguments.len() != parameter_count {
                return Err(ParseError {
                    kind: ParseErrorKind::InvalidFunctionParameters,
                    position: closing_position,
                });
            }
            return Ok(arguments);
        }
    }

    fn current(&self) -> &Token {
        &self.tokens[self.cursor]
    }

    fn peek_kind(&self, distance: usize) -> &TokenKind {
        &self.tokens[(self.cursor + distance).min(self.tokens.len() - 1)].kind
    }

    fn advance(&mut self) {
        if !matches!(self.current().kind, TokenKind::End) {
            self.cursor += 1;
        }
    }

    fn consume(&mut self, kind: &TokenKind) -> bool {
        if &self.current().kind == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(
        &mut self,
        kind: TokenKind,
        found_kind: ParseErrorKind,
        end_kind: ParseErrorKind,
    ) -> Result<(), ParseError> {
        if self.consume(&kind) {
            return Ok(());
        }
        Err(ParseError {
            kind: if matches!(self.current().kind, TokenKind::End) {
                end_kind
            } else {
                found_kind
            },
            position: self.current().position,
        })
    }
}

fn evaluate(
    expression: &Expression,
    variables: &[f64],
    inline_values: &[f64],
    use_degrees: bool,
    epsilon: f64,
    shared_values: &mut HashMap<usize, Result<f64, EvalError>>,
) -> Result<f64, EvalError> {
    match expression {
        Expression::Number(value) => Ok(*value),
        Expression::Variable(index) => Ok(variables[*index]),
        Expression::Inline(index) => Ok(inline_values[*index]),
        Expression::Shared(index, expression) => {
            if let Some(value) = shared_values.get(index) {
                return *value;
            }
            let value = evaluate(
                expression,
                variables,
                inline_values,
                use_degrees,
                epsilon,
                shared_values,
            );
            shared_values.insert(*index, value);
            value
        }
        Expression::Unary(operator, expression) => {
            let value = evaluate(
                expression,
                variables,
                inline_values,
                use_degrees,
                epsilon,
                shared_values,
            )?;
            Ok(match operator {
                UnaryOperator::Negate => -value,
                UnaryOperator::Not => f64::from(!truth(value)),
                UnaryOperator::ExactZero => {
                    if value == 0.0 {
                        f64::INFINITY
                    } else {
                        0.0
                    }
                }
                UnaryOperator::ExactNegative => {
                    if value < 0.0 {
                        f64::INFINITY
                    } else {
                        0.0
                    }
                }
            })
        }
        Expression::Binary(operator, left, right) => {
            let constant_exponent =
                matches!(operator, BinaryOperator::Power) && is_constant_expression(right);
            let left = evaluate(
                left,
                variables,
                inline_values,
                use_degrees,
                epsilon,
                shared_values,
            )?;
            let right = evaluate(
                right,
                variables,
                inline_values,
                use_degrees,
                epsilon,
                shared_values,
            )?;
            if *operator == BinaryOperator::Power && left == 0.0 && right < 0.0 {
                return Err(if constant_exponent {
                    EvalError::DivisionByZero
                } else {
                    EvalError::InvalidLogarithmOrPower
                });
            }
            evaluate_binary(*operator, left, right, epsilon)
        }
        Expression::Builtin(builtin, arguments) => {
            if *builtin == Builtin::If {
                let condition = evaluate(
                    &arguments[0],
                    variables,
                    inline_values,
                    use_degrees,
                    epsilon,
                    shared_values,
                )?;
                return evaluate(
                    if truth(condition) {
                        &arguments[1]
                    } else {
                        &arguments[2]
                    },
                    variables,
                    inline_values,
                    use_degrees,
                    epsilon,
                    shared_values,
                );
            }
            let values = arguments
                .iter()
                .map(|argument| {
                    evaluate(
                        argument,
                        variables,
                        inline_values,
                        use_degrees,
                        epsilon,
                        shared_values,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            if *builtin == Builtin::Pow && values[0] == 0.0 && values[1] < 0.0 {
                return Err(if is_constant_expression(&arguments[1]) {
                    EvalError::DivisionByZero
                } else {
                    EvalError::InvalidLogarithmOrPower
                });
            }
            evaluate_builtin(*builtin, &values, use_degrees, epsilon, true)
        }
        Expression::UserFunction(function, arguments) => {
            let values = arguments
                .iter()
                .map(|argument| {
                    evaluate(
                        argument,
                        variables,
                        inline_values,
                        use_degrees,
                        epsilon,
                        shared_values,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            match function {
                FunctionTarget::Callback(function) => Ok(function(&values)),
                FunctionTarget::Parser(parser) => parser.eval(&values),
            }
        }
    }
}

fn truth(value: f64) -> bool {
    value.abs() >= 0.5
}

fn is_constant_expression(expression: &Expression) -> bool {
    match expression {
        Expression::Number(_) => true,
        Expression::Unary(_, child) => is_constant_expression(child),
        Expression::Binary(_, left, right) => {
            is_constant_expression(left) && is_constant_expression(right)
        }
        Expression::Builtin(_, arguments) => arguments.iter().all(is_constant_expression),
        Expression::Shared(_, child) => is_constant_expression(child),
        Expression::Variable(_) | Expression::Inline(_) | Expression::UserFunction(_, _) => false,
    }
}

fn same_expression(left: &Expression, right: &Expression) -> bool {
    match (left, right) {
        (Expression::Number(left), Expression::Number(right)) => left == right,
        (Expression::Variable(left), Expression::Variable(right))
        | (Expression::Inline(left), Expression::Inline(right)) => left == right,
        (Expression::Shared(_, left), right) => same_expression(left, right),
        (left, Expression::Shared(_, right)) => same_expression(left, right),
        (Expression::Unary(left_operator, left), Expression::Unary(right_operator, right)) => {
            left_operator == right_operator && same_expression(left, right)
        }
        (
            Expression::Binary(left_operator, left_a, left_b),
            Expression::Binary(right_operator, right_a, right_b),
        ) => {
            left_operator == right_operator
                && same_expression(left_a, right_a)
                && same_expression(left_b, right_b)
        }
        (Expression::Builtin(left_builtin, left), Expression::Builtin(right_builtin, right)) => {
            left_builtin == right_builtin
                && left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_expression(left, right))
        }
        (
            Expression::UserFunction(left_target, left),
            Expression::UserFunction(right_target, right),
        ) => {
            let same_target = match (left_target, right_target) {
                (FunctionTarget::Callback(left), FunctionTarget::Callback(right)) => {
                    Arc::ptr_eq(left, right)
                }
                (FunctionTarget::Parser(left), FunctionTarget::Parser(right)) => {
                    Arc::ptr_eq(left, right)
                }
                _ => false,
            };
            same_target
                && left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| same_expression(left, right))
        }
        _ => false,
    }
}

fn collect_function_calls(expression: &Expression, calls: &mut Vec<(Expression, usize, usize)>) {
    match expression {
        Expression::Shared(_, child) | Expression::Unary(_, child) => {
            collect_function_calls(child, calls)
        }
        Expression::Binary(_, left, right) => {
            collect_function_calls(left, calls);
            collect_function_calls(right, calls);
        }
        Expression::Builtin(_, arguments) => {
            for argument in arguments {
                collect_function_calls(argument, calls);
            }
        }
        Expression::UserFunction(_, arguments) => {
            if let Some((_, _, count)) = calls
                .iter_mut()
                .find(|(previous, _, _)| same_expression(previous, expression))
            {
                *count += 1;
            } else {
                let index = calls.len();
                calls.push((expression.clone(), index, 1));
            }
            for argument in arguments {
                collect_function_calls(argument, calls);
            }
        }
        Expression::Number(_) | Expression::Variable(_) | Expression::Inline(_) => {}
    }
}

fn share_repeated_functions(
    expression: Expression,
    calls: &[(Expression, usize, usize)],
) -> Expression {
    match expression {
        Expression::Shared(_, child) => share_repeated_functions(*child, calls),
        Expression::Unary(operator, child) => {
            Expression::Unary(operator, Box::new(share_repeated_functions(*child, calls)))
        }
        Expression::Binary(operator, left, right) => Expression::Binary(
            operator,
            Box::new(share_repeated_functions(*left, calls)),
            Box::new(share_repeated_functions(*right, calls)),
        ),
        Expression::Builtin(builtin, arguments) => Expression::Builtin(
            builtin,
            arguments
                .into_iter()
                .map(|argument| share_repeated_functions(argument, calls))
                .collect(),
        ),
        Expression::UserFunction(function, arguments) => {
            let call = Expression::UserFunction(
                function,
                arguments
                    .into_iter()
                    .map(|argument| share_repeated_functions(argument, calls))
                    .collect(),
            );
            if let Some((_, index, count)) = calls
                .iter()
                .find(|(previous, _, _)| same_expression(previous, &call))
            {
                if *count > 1 {
                    Expression::Shared(*index, Box::new(call))
                } else {
                    call
                }
            } else {
                call
            }
        }
        expression => expression,
    }
}

fn is_trigonometric_square_sum(left: &Expression, right: &Expression) -> bool {
    fn square_argument(expression: &Expression, builtin: Builtin) -> Option<&Expression> {
        let base = match expression {
            Expression::Binary(BinaryOperator::Power, base, exponent) if matches!(exponent.as_ref(), Expression::Number(value) if *value == 2.0) => {
                base.as_ref()
            }
            Expression::Builtin(Builtin::Pow, arguments) if matches!(arguments.as_slice(), [_, Expression::Number(value)] if *value == 2.0) => {
                &arguments[0]
            }
            Expression::Binary(BinaryOperator::Multiply, left, right)
                if same_expression(left, right) =>
            {
                left.as_ref()
            }
            _ => return None,
        };
        match base {
            Expression::Builtin(found, arguments) if *found == builtin && arguments.len() == 1 => {
                Some(&arguments[0])
            }
            _ => None,
        }
    }

    matches!(
        (
            square_argument(left, Builtin::Sin),
            square_argument(right, Builtin::Cos)
        ),
        (Some(left_sin), Some(right_cos)) if same_expression(left_sin, right_cos)
    ) || matches!(
        (
            square_argument(left, Builtin::Cos),
            square_argument(right, Builtin::Sin)
        ),
        (Some(left_cos), Some(right_sin)) if same_expression(left_cos, right_sin)
    )
}

fn rewrite_compile_patterns(expression: Expression) -> Expression {
    match expression {
        Expression::Shared(index, child) => {
            Expression::Shared(index, Box::new(rewrite_compile_patterns(*child)))
        }
        Expression::Unary(operator, child) => {
            Expression::Unary(operator, Box::new(rewrite_compile_patterns(*child)))
        }
        Expression::Binary(operator, left, right) => {
            let left = rewrite_compile_patterns(*left);
            let right = rewrite_compile_patterns(*right);
            if operator == BinaryOperator::Divide
                && matches!(
                    (&left, &right),
                    (Expression::Variable(left), Expression::Variable(right)) if left == right
                )
            {
                Expression::Number(1.0)
            } else {
                Expression::Binary(operator, Box::new(left), Box::new(right))
            }
        }
        Expression::Builtin(builtin, arguments) => {
            let arguments = arguments
                .into_iter()
                .map(rewrite_compile_patterns)
                .collect::<Vec<_>>();
            if let [Expression::Builtin(exponential, inner_arguments)] = arguments.as_slice() {
                if let [argument] = inner_arguments.as_slice() {
                    if let Some(rewritten) = rewrite_log_exp(builtin, *exponential, argument) {
                        return rewritten;
                    }
                    if builtin == Builtin::Sinh && *exponential == Builtin::Asinh {
                        return argument.clone();
                    }
                    if builtin == Builtin::Sinh && *exponential == Builtin::Acosh {
                        if let Expression::Builtin(Builtin::Log10, log_arguments) = argument {
                            if let [argument @ Expression::Variable(_)] = log_arguments.as_slice() {
                                return rewrite_hyperbolic_acosh_log10(Builtin::Sinh, argument);
                            }
                        }
                        return rewrite_sinh_acosh(argument.clone());
                    }
                }
            }
            if let [argument @ Expression::Builtin(inner, _)] = arguments.as_slice() {
                let redundant_abs = builtin == Builtin::Abs && *inner == Builtin::Sqrt;
                let redundant_rounding = builtin == Builtin::Int
                    && matches!(
                        inner,
                        Builtin::Ceil | Builtin::Floor | Builtin::Int | Builtin::Trunc
                    );
                if redundant_abs || redundant_rounding {
                    return argument.clone();
                }
            }
            Expression::Builtin(builtin, arguments)
        }
        Expression::UserFunction(function, arguments) => Expression::UserFunction(
            function,
            arguments
                .into_iter()
                .map(rewrite_compile_patterns)
                .collect(),
        ),
        other => other,
    }
}

fn rewrite_log_exp(
    logarithm: Builtin,
    exponential: Builtin,
    argument: &Expression,
) -> Option<Expression> {
    let coefficient = match (logarithm, exponential) {
        (Builtin::Log, Builtin::Exp) | (Builtin::Log2, Builtin::Exp2) => {
            return Some(argument.clone());
        }
        (Builtin::Log, Builtin::Exp2) => std::f64::consts::LN_2,
        (Builtin::Log2, Builtin::Exp) => std::f64::consts::LOG2_E,
        (Builtin::Log10, Builtin::Exp) => std::f64::consts::LOG10_E,
        (Builtin::Log10, Builtin::Exp2) => std::f64::consts::LOG10_2,
        _ => return None,
    };
    Some(Expression::Binary(
        BinaryOperator::Multiply,
        Box::new(argument.clone()),
        Box::new(Expression::Number(coefficient)),
    ))
}

fn rewrite_exp_log(outer: Builtin, inner: Builtin, argument: &Expression) -> Option<Expression> {
    if (outer, inner) == (Builtin::Exp, Builtin::Log) {
        return Some(argument.clone());
    }
    let exponent = exp_log_power_exponent(outer, inner)?;
    Some(Expression::Binary(
        BinaryOperator::Power,
        Box::new(argument.clone()),
        Box::new(Expression::Number(exponent)),
    ))
}

fn exp_log_power_exponent(outer: Builtin, inner: Builtin) -> Option<f64> {
    match (outer, inner) {
        (Builtin::Exp, Builtin::Log2) => Some(std::f64::consts::LOG2_E),
        (Builtin::Exp, Builtin::Log10) => Some(LEGACY_OPTIMIZER_LOG10_E),
        (Builtin::Exp2, Builtin::Log) => Some(std::f64::consts::LN_2),
        (Builtin::Exp2, Builtin::Log10) => Some(LEGACY_OPTIMIZER_LOG10_2),
        _ => None,
    }
}

fn rewrite_sixth_root(outer: Builtin, inner: Builtin, argument: &Expression) -> Option<Expression> {
    if !matches!(
        (outer, inner),
        (Builtin::Sqrt, Builtin::Cbrt) | (Builtin::Cbrt, Builtin::Sqrt)
    ) {
        return None;
    }
    Some(Expression::Binary(
        BinaryOperator::Power,
        Box::new(argument.clone()),
        Box::new(Expression::Number(1.0 / 6.0)),
    ))
}

fn rewrite_signed_power(argument: &Expression, exponent: f64) -> Expression {
    let magnitude = || {
        Expression::Binary(
            BinaryOperator::Power,
            Box::new(Expression::Builtin(Builtin::Abs, vec![argument.clone()])),
            Box::new(Expression::Number(exponent)),
        )
    };
    Expression::Builtin(
        Builtin::If,
        vec![
            Expression::Binary(
                BinaryOperator::Less,
                Box::new(argument.clone()),
                Box::new(Expression::Number(0.0)),
            ),
            Expression::Unary(UnaryOperator::Negate, Box::new(magnitude())),
            magnitude(),
        ],
    )
}

fn rewrite_legacy_zero_log_error(argument: &Expression, fallback: &Expression) -> Expression {
    let is_zero = Expression::Unary(UnaryOperator::ExactZero, Box::new(argument.clone()));
    Expression::Builtin(
        Builtin::If,
        vec![
            is_zero,
            Expression::Builtin(
                Builtin::Log,
                vec![Expression::Builtin(Builtin::Abs, vec![argument.clone()])],
            ),
            fallback.clone(),
        ],
    )
}

fn rewrite_legacy_negative_sqrt_log_error(
    expression: &Expression,
    argument: &Expression,
) -> Expression {
    let is_negative = Expression::Unary(UnaryOperator::ExactNegative, Box::new(argument.clone()));
    let magnitude = Expression::Builtin(Builtin::Abs, vec![argument.clone()]);
    let zero = Expression::Binary(
        BinaryOperator::Add,
        Box::new(magnitude),
        Box::new(argument.clone()),
    );
    Expression::Builtin(
        Builtin::If,
        vec![
            is_negative,
            Expression::Builtin(Builtin::Log, vec![zero]),
            expression.clone(),
        ],
    )
}

fn rewrite_nested_optimizer_expression(expression: &Expression) -> Option<Expression> {
    let Expression::Builtin(outer, arguments) = expression else {
        return None;
    };
    let inverse_exponential_inner = match outer {
        Builtin::Csc => Some(Builtin::Asin),
        Builtin::Sec => Some(Builtin::Acos),
        _ => None,
    };
    if let Some(inner) = inverse_exponential_inner {
        if let [Expression::Builtin(actual_inner, inner_arguments)] = arguments.as_slice() {
            if *actual_inner == inner {
                if let [argument] = inner_arguments.as_slice() {
                    if let Some(rewritten) = rewrite_inverse_exponential(argument) {
                        return Some(rewritten);
                    }
                }
            }
        }
    }
    if matches!(outer, Builtin::Log | Builtin::Log2 | Builtin::Log10) {
        if let [Expression::Builtin(Builtin::Csc, csc_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Acos, acos_arguments)] = csc_arguments.as_slice() {
                if let [argument @ Expression::Variable(_)] = acos_arguments.as_slice() {
                    let log_sine = Expression::Builtin(
                        Builtin::Log,
                        vec![Expression::Builtin(
                            Builtin::Sin,
                            vec![Expression::Builtin(Builtin::Acos, vec![argument.clone()])],
                        )],
                    );
                    let factor = match outer {
                        Builtin::Log => {
                            return Some(Expression::Unary(
                                UnaryOperator::Negate,
                                Box::new(log_sine),
                            ))
                        }
                        Builtin::Log2 => -std::f64::consts::LOG2_E,
                        Builtin::Log10 => -std::f64::consts::LOG10_E,
                        _ => unreachable!(),
                    };
                    return Some(Expression::Binary(
                        BinaryOperator::Multiply,
                        Box::new(log_sine),
                        Box::new(Expression::Number(factor)),
                    ));
                }
            }
        }
    }
    if let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() {
        if matches!(
            (*outer, *inner),
            (Builtin::Cbrt, Builtin::Cbrt | Builtin::Sqrt) | (Builtin::Sqrt, Builtin::Cbrt)
        ) {
            if let [Expression::Builtin(Builtin::Acos, acos_arguments)] = inner_arguments.as_slice()
            {
                if let [Expression::Variable(_)] = acos_arguments.as_slice() {
                    let leaf = Expression::Builtin(Builtin::Acos, acos_arguments.clone());
                    return Some(rewrite_legacy_zero_log_error(&leaf, expression));
                }
            }
        }
    }
    if *outer == Builtin::Cot {
        if let [Expression::Builtin(Builtin::Abs, abs_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Acos, acos_arguments)] = abs_arguments.as_slice() {
                if let [argument @ Expression::Variable(_)] = acos_arguments.as_slice() {
                    let tangent = Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression::Builtin(
                            Builtin::Sin,
                            vec![Expression::Builtin(Builtin::Acos, vec![argument.clone()])],
                        )),
                        Box::new(argument.clone()),
                    );
                    return Some(Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression::Number(1.0)),
                        Box::new(tangent),
                    ));
                }
            }
        }
    }
    if matches!(outer, Builtin::Cot | Builtin::Tan) {
        if let [Expression::Builtin(Builtin::Asin, asin_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Tanh, tanh_arguments)] = asin_arguments.as_slice()
            {
                if let [argument @ Expression::Variable(_)] = tanh_arguments.as_slice() {
                    let tangent_numerator =
                        Expression::Builtin(Builtin::Sinh, vec![argument.clone()]);
                    let tangent_denominator = Expression::Binary(
                        BinaryOperator::Multiply,
                        Box::new(Expression::Builtin(Builtin::Cosh, vec![argument.clone()])),
                        Box::new(Expression::Builtin(
                            Builtin::Cos,
                            vec![Expression::Builtin(
                                Builtin::Asin,
                                vec![Expression::Builtin(Builtin::Tanh, vec![argument.clone()])],
                            )],
                        )),
                    );
                    let (numerator, denominator) = if *outer == Builtin::Tan {
                        (tangent_numerator, tangent_denominator)
                    } else {
                        (tangent_denominator, tangent_numerator)
                    };
                    return Some(Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(numerator),
                        Box::new(denominator),
                    ));
                }
            }
        }
    }
    if matches!(outer, Builtin::Exp | Builtin::Exp2) {
        if let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() {
            if matches!(inner, Builtin::Log | Builtin::Log2 | Builtin::Log10) {
                if let [argument @ Expression::Builtin(Builtin::Acos, _)] =
                    inner_arguments.as_slice()
                {
                    if let Some(fallback) = rewrite_exp_log(*outer, *inner, argument) {
                        if !matches!((*outer, *inner), (Builtin::Exp, Builtin::Log)) {
                            return Some(rewrite_legacy_zero_log_error(argument, &fallback));
                        }
                    }
                }
            }
        }
    }
    if *outer == Builtin::Cbrt {
        if let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() {
            let inverse_trig_argument = matches!(
                inner_arguments.as_slice(),
                [Expression::Builtin(Builtin::Acos | Builtin::Asin, _)]
            );
            if !inverse_trig_argument {
                let reciprocal = match inner {
                    Builtin::Cot => Some(Builtin::Tan),
                    Builtin::Csc => Some(Builtin::Sin),
                    Builtin::Sec => Some(Builtin::Cos),
                    _ => None,
                };
                if let Some(reciprocal) = reciprocal {
                    return Some(Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression::Number(1.0)),
                        Box::new(Expression::Builtin(
                            Builtin::Cbrt,
                            vec![Expression::Builtin(reciprocal, inner_arguments.clone())],
                        )),
                    ));
                }
            }
        }
    }
    if !matches!(outer, Builtin::Exp | Builtin::Exp2) {
        if let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() {
            if matches!(inner, Builtin::Log | Builtin::Log2 | Builtin::Log10) {
                if let [Expression::Builtin(Builtin::Log10, log10_arguments)] =
                    inner_arguments.as_slice()
                {
                    if let [argument @ Expression::Variable(_)] = log10_arguments.as_slice() {
                        let base_coefficient = match inner {
                            Builtin::Log => 1.0,
                            Builtin::Log2 => std::f64::consts::LOG2_E,
                            Builtin::Log10 => std::f64::consts::LOG10_E,
                            _ => unreachable!(),
                        };
                        let coefficient = std::f64::consts::LOG10_E.ln() * base_coefficient;
                        if *inner == Builtin::Log2 && matches!(outer, Builtin::Cbrt | Builtin::Sqrt)
                        {
                            let adjusted_log = Expression::Binary(
                                BinaryOperator::Add,
                                Box::new(Expression::Number(std::f64::consts::LOG10_E.ln())),
                                Box::new(Expression::Builtin(
                                    Builtin::Log,
                                    vec![Expression::Builtin(Builtin::Log, vec![argument.clone()])],
                                )),
                            );
                            let scale = if *outer == Builtin::Cbrt {
                                std::f64::consts::LOG2_E.cbrt()
                            } else {
                                std::f64::consts::LOG2_E.sqrt()
                            };
                            return Some(Expression::Binary(
                                BinaryOperator::Multiply,
                                Box::new(Expression::Number(scale)),
                                Box::new(Expression::Builtin(*outer, vec![adjusted_log])),
                            ));
                        }
                        let logarithm = Expression::Builtin(
                            Builtin::Log,
                            vec![Expression::Builtin(Builtin::Log, vec![argument.clone()])],
                        );
                        let logarithm = if *inner == Builtin::Log {
                            logarithm
                        } else {
                            Expression::Binary(
                                BinaryOperator::Multiply,
                                Box::new(logarithm),
                                Box::new(Expression::Number(base_coefficient)),
                            )
                        };
                        return Some(Expression::Builtin(
                            *outer,
                            vec![Expression::Binary(
                                BinaryOperator::Add,
                                Box::new(Expression::Number(coefficient)),
                                Box::new(logarithm),
                            )],
                        ));
                    }
                }
            }
        }
    }
    if let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() {
        if let [Expression::Builtin(innermost, innermost_arguments)] = inner_arguments.as_slice() {
            let zero_log_error = matches!(
                (*outer, *inner, *innermost),
                (
                    Builtin::Cbrt,
                    Builtin::Cbrt,
                    Builtin::Abs | Builtin::Cot | Builtin::Csc
                ) | (
                    Builtin::Cbrt,
                    Builtin::Sqrt,
                    Builtin::Abs | Builtin::Cot | Builtin::Csc
                ) | (
                    Builtin::Sqrt,
                    Builtin::Cbrt,
                    Builtin::Abs | Builtin::Cot | Builtin::Csc
                ) | (
                    Builtin::Exp2,
                    Builtin::Log | Builtin::Log10,
                    Builtin::Abs | Builtin::Cot | Builtin::Csc
                ) | (
                    Builtin::Exp,
                    Builtin::Log2 | Builtin::Log10,
                    Builtin::Abs | Builtin::Cot | Builtin::Csc
                ) | (
                    Builtin::Log | Builtin::Log2 | Builtin::Log10,
                    Builtin::Abs,
                    Builtin::Cot | Builtin::Csc
                )
            );
            if zero_log_error {
                if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
                    let fallback = if (*outer, *inner) == (Builtin::Cbrt, Builtin::Sqrt) {
                        let leaf = Expression::Builtin(*innermost, innermost_arguments.clone());
                        rewrite_sixth_root(*outer, *inner, &leaf)?
                    } else if (*outer, *inner) == (Builtin::Sqrt, Builtin::Cbrt)
                        && matches!(*innermost, Builtin::Cot | Builtin::Csc)
                    {
                        let leaf = Expression::Builtin(*innermost, innermost_arguments.clone());
                        rewrite_signed_power(&leaf, 1.0 / 6.0)
                    } else if matches!(*innermost, Builtin::Cot | Builtin::Csc) {
                        if let Some(exponent) = exp_log_power_exponent(*outer, *inner) {
                            let leaf = Expression::Builtin(*innermost, innermost_arguments.clone());
                            rewrite_signed_power(&leaf, exponent)
                        } else {
                            expression.clone()
                        }
                    } else {
                        expression.clone()
                    };
                    return Some(rewrite_legacy_zero_log_error(argument, &fallback));
                }
            }
            if matches!(*outer, Builtin::Cos | Builtin::Cosh)
                && (*inner, *innermost) == (Builtin::Abs, Builtin::Tanh)
            {
                if let [argument] = innermost_arguments.as_slice() {
                    return Some(Expression::Builtin(
                        *outer,
                        vec![Expression::Builtin(
                            Builtin::Tanh,
                            vec![Expression::Builtin(Builtin::Abs, vec![argument.clone()])],
                        )],
                    ));
                }
            }
            if (*outer, *inner, *innermost) == (Builtin::Cosh, Builtin::Asinh, Builtin::Sinh) {
                if let [argument] = innermost_arguments.as_slice() {
                    return Some(Expression::Builtin(Builtin::Cosh, vec![argument.clone()]));
                }
            }
        }
    }
    if matches!(*outer, Builtin::Log | Builtin::Log2 | Builtin::Log10) {
        if let [Expression::Builtin(Builtin::Sqrt, sqrt_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(
                leaf @ (Builtin::Cot | Builtin::Csc | Builtin::Sec),
                leaf_arguments,
            )] = sqrt_arguments.as_slice()
            {
                if matches!(leaf_arguments.as_slice(), [Expression::Variable(_)]) {
                    let argument = Expression::Builtin(*leaf, leaf_arguments.clone());
                    return Some(rewrite_legacy_negative_sqrt_log_error(
                        expression, &argument,
                    ));
                }
            }
        }
    }
    if *outer == Builtin::Cosh {
        if let [Expression::Builtin(Builtin::Asinh, asinh_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Sqrt, sqrt_arguments)] = asinh_arguments.as_slice()
            {
                if let [argument @ Expression::Variable(_)] = sqrt_arguments.as_slice() {
                    return Some(Expression::Builtin(
                        Builtin::Sqrt,
                        vec![Expression::Binary(
                            BinaryOperator::Add,
                            Box::new(Expression::Number(1.0)),
                            Box::new(argument.clone()),
                        )],
                    ));
                }
            }
        }
    }
    if *outer == Builtin::Sqrt {
        if let [Expression::Builtin(Builtin::Csc, csc_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Asin, asin_arguments)] = csc_arguments.as_slice() {
                if let [argument @ Expression::Variable(_)] = asin_arguments.as_slice() {
                    return Some(Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression::Number(1.0)),
                        Box::new(Expression::Binary(
                            BinaryOperator::Power,
                            Box::new(argument.clone()),
                            Box::new(Expression::Number(0.5)),
                        )),
                    ));
                }
            }
        }
        if let [Expression::Builtin(Builtin::Sec, sec_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Acos, acos_arguments)] = sec_arguments.as_slice() {
                if let [argument @ Expression::Variable(_)] = acos_arguments.as_slice() {
                    return Some(Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(Expression::Number(1.0)),
                        Box::new(Expression::Binary(
                            BinaryOperator::Power,
                            Box::new(argument.clone()),
                            Box::new(Expression::Number(0.5)),
                        )),
                    ));
                }
            }
        }
        if let Some(rewritten) = rewrite_sqrt_bounded_acosh(arguments) {
            return Some(rewritten);
        }
        if let [Expression::Builtin(Builtin::Sqrt, middle_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Cbrt, inner_arguments)] =
                middle_arguments.as_slice()
            {
                if let [argument @ Expression::Variable(_)] = inner_arguments.as_slice() {
                    return Some(rewrite_signed_power(argument, 1.0 / 12.0));
                }
            }
        }
        if let [Expression::Builtin(Builtin::Cbrt, middle_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Sqrt, inner_arguments)] =
                middle_arguments.as_slice()
            {
                if let [argument @ Expression::Variable(_)] = inner_arguments.as_slice() {
                    return Some(rewrite_signed_power(argument, 1.0 / 12.0));
                }
            }
        }
        if let [Expression::Builtin(exponential, exponential_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(logarithm, logarithm_arguments)] =
                exponential_arguments.as_slice()
            {
                if let [argument @ Expression::Variable(_)] = logarithm_arguments.as_slice() {
                    if let Some(exponent) = exp_log_power_exponent(*exponential, *logarithm) {
                        return Some(rewrite_signed_power(argument, exponent / 2.0));
                    }
                }
            }
        }
    }
    if *outer == Builtin::Cbrt {
        if let [Expression::Builtin(Builtin::Sqrt, sqrt_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Sqrt, nested_arguments)] =
                sqrt_arguments.as_slice()
            {
                if let [argument @ Expression::Variable(_)] = nested_arguments.as_slice() {
                    return Some(Expression::Binary(
                        BinaryOperator::Power,
                        Box::new(argument.clone()),
                        Box::new(Expression::Number(1.0 / 12.0)),
                    ));
                }
            }
            if let [argument] = sqrt_arguments.as_slice() {
                return rewrite_sixth_root(Builtin::Cbrt, Builtin::Sqrt, argument);
            }
        }
    }
    if *outer == Builtin::Abs {
        if let [Expression::Builtin(Builtin::Cbrt, cbrt_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(Builtin::Sqrt, sqrt_arguments)] = cbrt_arguments.as_slice()
            {
                if let [argument] = sqrt_arguments.as_slice() {
                    return rewrite_sixth_root(Builtin::Cbrt, Builtin::Sqrt, argument);
                }
            }
        }
        if let [Expression::Builtin(
            inner @ (Builtin::Asinh
            | Builtin::Atan
            | Builtin::Atanh
            | Builtin::Ceil
            | Builtin::Floor
            | Builtin::Int
            | Builtin::Sinh
            | Builtin::Tanh
            | Builtin::Trunc),
            inner_arguments,
        )] = arguments.as_slice()
        {
            if let [Expression::Builtin(Builtin::Cbrt, cbrt_arguments)] = inner_arguments.as_slice()
            {
                if matches!(cbrt_arguments.as_slice(), [Expression::Variable(_)]) {
                    return Some(Expression::Builtin(*inner, inner_arguments.clone()));
                }
            }
            if let [Expression::Builtin(Builtin::Sqrt, sqrt_arguments)] = inner_arguments.as_slice()
            {
                if matches!(sqrt_arguments.as_slice(), [Expression::Variable(_)])
                    && matches!(
                        *inner,
                        Builtin::Atan
                            | Builtin::Ceil
                            | Builtin::Floor
                            | Builtin::Sinh
                            | Builtin::Tanh
                            | Builtin::Trunc
                    )
                {
                    return Some(Expression::Builtin(*inner, inner_arguments.clone()));
                }
            }
        }
        if let [Expression::Builtin(exponential, exponential_arguments)] = arguments.as_slice() {
            if let [Expression::Builtin(logarithm, logarithm_arguments)] =
                exponential_arguments.as_slice()
            {
                if let [argument] = logarithm_arguments.as_slice() {
                    if (*exponential, *logarithm) == (Builtin::Exp2, Builtin::Log2) {
                        return Some(argument.clone());
                    }
                    if let Some(rewritten) = rewrite_exp_log(*exponential, *logarithm, argument) {
                        return Some(rewritten);
                    }
                }
            }
        }
    }
    let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() else {
        return None;
    };
    let [Expression::Builtin(innermost, innermost_arguments)] = inner_arguments.as_slice() else {
        return None;
    };
    if matches!(
        (*outer, *inner, *innermost),
        (
            Builtin::Sinh | Builtin::Tanh,
            Builtin::Acosh,
            Builtin::Log10
        )
    ) {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            return Some(rewrite_hyperbolic_acosh_log10(*outer, argument));
        }
    }
    if matches!(
        (*outer, *inner, *innermost),
        (Builtin::Cot, Builtin::Atan, Builtin::Sqrt) | (Builtin::Csc, Builtin::Asin, Builtin::Sqrt)
    ) {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            // The power path preserves the optimizer's NaN-without-error result.
            return Some(Expression::Binary(
                BinaryOperator::Divide,
                Box::new(Expression::Number(1.0)),
                Box::new(Expression::Binary(
                    BinaryOperator::Power,
                    Box::new(argument.clone()),
                    Box::new(Expression::Number(0.5)),
                )),
            ));
        }
    }
    if (*outer, *inner, *innermost) == (Builtin::Sqrt, Builtin::Cot, Builtin::Acos) {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            let rewritten = Expression::Binary(
                BinaryOperator::Power,
                Box::new(Expression::Builtin(
                    Builtin::Cot,
                    vec![Expression::Builtin(Builtin::Acos, vec![argument.clone()])],
                )),
                Box::new(Expression::Number(0.5)),
            );
            return Some(Expression::Builtin(
                Builtin::If,
                vec![
                    Expression::Unary(UnaryOperator::ExactZero, Box::new(argument.clone())),
                    Expression::Binary(
                        BinaryOperator::Divide,
                        Box::new(argument.clone()),
                        Box::new(Expression::Number(0.0)),
                    ),
                    rewritten,
                ],
            ));
        }
    }
    if *outer == Builtin::Atan && *inner == Builtin::Tan {
        if let [argument @ Expression::Builtin(Builtin::Acos, _)] = inner_arguments.as_slice() {
            return Some(argument.clone());
        }
    }
    if *outer == Builtin::Sec && *inner == Builtin::Abs {
        if let [Expression::Builtin(Builtin::Acos, acos_arguments)] = inner_arguments.as_slice() {
            if let [argument] = acos_arguments.as_slice() {
                return Some(Expression::Binary(
                    BinaryOperator::Divide,
                    Box::new(Expression::Number(1.0)),
                    Box::new(argument.clone()),
                ));
            }
        }
    }
    if (*outer, *inner, *innermost) == (Builtin::Sec, Builtin::Acos, Builtin::Sqrt) {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            return Some(Expression::Binary(
                BinaryOperator::Divide,
                Box::new(Expression::Number(1.0)),
                Box::new(Expression::Binary(
                    BinaryOperator::Power,
                    Box::new(argument.clone()),
                    Box::new(Expression::Number(0.5)),
                )),
            ));
        }
    }
    if *outer == Builtin::Tan && *inner == Builtin::Abs && *innermost == Builtin::Acos {
        if let [argument] = innermost_arguments.as_slice() {
            return rewrite_tan_cot_acos(Builtin::Tan, Builtin::Acos, argument);
        }
    }
    if *outer == Builtin::Cos && *inner == Builtin::Abs && *innermost == Builtin::Acos {
        if let [argument] = innermost_arguments.as_slice() {
            return Some(argument.clone());
        }
    }
    if *outer == Builtin::Cot && *inner == Builtin::Abs && *innermost == Builtin::Acos {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            return Some(Expression::Binary(
                BinaryOperator::Divide,
                Box::new(Expression::Number(1.0)),
                Box::new(Expression::Builtin(
                    Builtin::Cot,
                    vec![Expression::Builtin(Builtin::Asin, vec![argument.clone()])],
                )),
            ));
        }
    }
    if *innermost == Builtin::Sqrt {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            if let Some(exponent) = exp_log_power_exponent(*outer, *inner) {
                return Some(Expression::Binary(
                    BinaryOperator::Power,
                    Box::new(argument.clone()),
                    Box::new(Expression::Number(exponent / 2.0)),
                ));
            }
        }
    }
    let reciprocal = match (*outer, inner, innermost) {
        (Builtin::Sec, Builtin::Acos, Builtin::Cot) => Some(Builtin::Tan),
        (Builtin::Sec, Builtin::Acos, Builtin::Csc) => Some(Builtin::Sin),
        (Builtin::Cot, Builtin::Atan, Builtin::Cot)
        | (Builtin::Csc, Builtin::Asin, Builtin::Cot) => Some(Builtin::Tan),
        (Builtin::Cot, Builtin::Atan, Builtin::Csc)
        | (Builtin::Csc, Builtin::Asin, Builtin::Csc) => Some(Builtin::Sin),
        _ => None,
    };
    if let Some(reciprocal) = reciprocal {
        if let [argument] = innermost_arguments.as_slice() {
            return Some(Expression::Builtin(reciprocal, vec![argument.clone()]));
        }
    }
    let log_exponential_root_factor = match (*outer, *inner, *innermost) {
        (Builtin::Log, Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt, Builtin::Exp) => Some(1.0),
        (Builtin::Log, Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt, Builtin::Exp2) => {
            Some(std::f64::consts::LN_2)
        }
        (Builtin::Log2, Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt, Builtin::Exp) => {
            Some(std::f64::consts::LOG2_E)
        }
        (Builtin::Log2, Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt, Builtin::Exp2) => Some(1.0),
        (Builtin::Log10, Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt, Builtin::Exp) => {
            Some(std::f64::consts::LOG10_E)
        }
        (Builtin::Log10, Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt, Builtin::Exp2) => {
            Some(std::f64::consts::LOG10_2)
        }
        _ => None,
    };
    if let Some(factor) = log_exponential_root_factor {
        let root_factor = match *inner {
            Builtin::Abs => 1.0,
            Builtin::Cbrt => 1.0 / 3.0,
            Builtin::Sqrt => 0.5,
            _ => unreachable!(),
        };
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            return Some(Expression::Binary(
                BinaryOperator::Multiply,
                Box::new(argument.clone()),
                Box::new(Expression::Number(factor * root_factor)),
            ));
        }
    }
    let negative_zero_log = matches!(
        (*outer, *inner, *innermost),
        (
            Builtin::Log | Builtin::Log2 | Builtin::Log10,
            Builtin::Sec,
            Builtin::Asin | Builtin::Atan | Builtin::Sin | Builtin::Tanh,
        ) | (
            Builtin::Log | Builtin::Log2 | Builtin::Log10,
            Builtin::Csc,
            Builtin::Acos
        ) | (
            Builtin::Log | Builtin::Log2 | Builtin::Log10,
            Builtin::Abs,
            Builtin::Sec
        )
    );
    if negative_zero_log {
        if let [argument @ Expression::Variable(_)] = innermost_arguments.as_slice() {
            let near_zero = Expression::Binary(
                BinaryOperator::Equal,
                Box::new(Expression::Binary(
                    BinaryOperator::Power,
                    Box::new(argument.clone()),
                    Box::new(Expression::Number(2.0)),
                )),
                Box::new(Expression::Number(0.0)),
            );
            return Some(Expression::Builtin(
                Builtin::If,
                vec![near_zero, Expression::Number(-0.0), expression.clone()],
            ));
        }
    }
    let value = match (*outer, inner, innermost) {
        (
            Builtin::Acos,
            Builtin::Sec,
            Builtin::Asin | Builtin::Atan | Builtin::Cos | Builtin::Sin | Builtin::Tanh,
        ) => 0.0,
        (
            Builtin::Asin,
            Builtin::Sec,
            Builtin::Asin | Builtin::Atan | Builtin::Cos | Builtin::Sin | Builtin::Tanh,
        ) => std::f64::consts::FRAC_PI_2,
        (Builtin::Acos, Builtin::Csc, Builtin::Acos) => 0.0,
        (Builtin::Asin, Builtin::Csc, Builtin::Acos) => std::f64::consts::FRAC_PI_2,
        (
            Builtin::Acos,
            Builtin::Exp | Builtin::Exp2,
            Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt,
        ) => 0.0,
        (
            Builtin::Asin,
            Builtin::Exp | Builtin::Exp2,
            Builtin::Abs | Builtin::Cbrt | Builtin::Sqrt,
        ) => std::f64::consts::FRAC_PI_2,
        (
            Builtin::Acos,
            Builtin::Abs
            | Builtin::Cbrt
            | Builtin::Ceil
            | Builtin::Floor
            | Builtin::Int
            | Builtin::Sqrt
            | Builtin::Trunc,
            Builtin::Cosh,
        ) => 0.0,
        (
            Builtin::Asin,
            Builtin::Abs
            | Builtin::Cbrt
            | Builtin::Ceil
            | Builtin::Floor
            | Builtin::Int
            | Builtin::Sqrt
            | Builtin::Trunc,
            Builtin::Cosh,
        ) => std::f64::consts::FRAC_PI_2,
        (Builtin::Atan, Builtin::Atanh, Builtin::Cosh) => std::f64::consts::FRAC_PI_2,
        (Builtin::Tanh, Builtin::Atanh, Builtin::Cosh) => 1.0,
        (Builtin::Tanh, Builtin::Acosh, Builtin::Sin | Builtin::Cos | Builtin::Tanh) => 0.0,
        _ => return None,
    };
    Some(Expression::Number(value))
}

fn rewrite_sqrt_bounded_acosh(arguments: &[Expression]) -> Option<Expression> {
    let [Expression::Binary(BinaryOperator::Subtract, radicand, one)] = arguments else {
        return None;
    };
    let Expression::Number(one) = one.as_ref() else {
        return None;
    };
    if *one != 1.0 {
        return None;
    }
    let Expression::Binary(BinaryOperator::Power, base, exponent) = radicand.as_ref() else {
        return None;
    };
    let Expression::Number(exponent) = exponent.as_ref() else {
        return None;
    };
    if *exponent != 2.0
        || !matches!(
            base.as_ref(),
            Expression::Builtin(Builtin::Sin | Builtin::Cos | Builtin::Tanh, _)
        )
    {
        return None;
    }
    Some(Expression::Number(0.0))
}

fn rewrite_sqrt_reciprocal(inner: Builtin, angle: &Expression) -> Option<Expression> {
    let reciprocal = match inner {
        Builtin::Cot => Builtin::Tan,
        Builtin::Csc => Builtin::Sin,
        Builtin::Sec => Builtin::Cos,
        _ => return None,
    };
    Some(Expression::Binary(
        BinaryOperator::Power,
        Box::new(Expression::Builtin(reciprocal, vec![angle.clone()])),
        Box::new(Expression::Number(-0.5)),
    ))
}

fn rewrite_inverse_exponential(expression: &Expression) -> Option<Expression> {
    let Expression::Builtin(exponential, arguments) = expression else {
        return None;
    };
    let [argument] = arguments.as_slice() else {
        return None;
    };
    let exponent = match exponential {
        Builtin::Exp => Expression::Unary(UnaryOperator::Negate, Box::new(argument.clone())),
        Builtin::Exp2 => Expression::Binary(
            BinaryOperator::Multiply,
            Box::new(Expression::Number(-std::f64::consts::LN_2)),
            Box::new(argument.clone()),
        ),
        _ => return None,
    };
    Some(Expression::Builtin(Builtin::Exp, vec![exponent]))
}

fn rewrite_tan_cot_acos(
    outer: Builtin,
    inner: Builtin,
    argument: &Expression,
) -> Option<Expression> {
    if !matches!(outer, Builtin::Tan | Builtin::Cot) || inner != Builtin::Acos {
        return None;
    }
    if let Some(reciprocal_exponential) = rewrite_inverse_exponential(argument) {
        let tangent = Expression::Binary(
            BinaryOperator::Multiply,
            Box::new(reciprocal_exponential),
            Box::new(Expression::Builtin(
                Builtin::Sin,
                vec![Expression::Builtin(Builtin::Acos, vec![argument.clone()])],
            )),
        );
        return Some(if outer == Builtin::Cot {
            Expression::Binary(
                BinaryOperator::Divide,
                Box::new(Expression::Number(1.0)),
                Box::new(tangent),
            )
        } else {
            tangent
        });
    }
    let sine = Expression::Builtin(
        Builtin::Sin,
        vec![Expression::Builtin(Builtin::Acos, vec![argument.clone()])],
    );
    let tangent = Expression::Binary(
        BinaryOperator::Divide,
        Box::new(sine),
        Box::new(argument.clone()),
    );
    Some(if outer == Builtin::Cot {
        Expression::Binary(
            BinaryOperator::Divide,
            Box::new(Expression::Number(1.0)),
            Box::new(tangent),
        )
    } else {
        tangent
    })
}

fn rewrite_sinh_acosh(argument: Expression) -> Expression {
    Expression::Builtin(
        Builtin::Sqrt,
        vec![Expression::Binary(
            BinaryOperator::Subtract,
            Box::new(Expression::Binary(
                BinaryOperator::Power,
                Box::new(argument),
                Box::new(Expression::Number(2.0)),
            )),
            Box::new(Expression::Number(1.0)),
        )],
    )
}

fn rewrite_hyperbolic_acosh_log10(outer: Builtin, argument: &Expression) -> Expression {
    let log_squared = Expression::Binary(
        BinaryOperator::Power,
        Box::new(Expression::Builtin(Builtin::Log, vec![argument.clone()])),
        Box::new(Expression::Number(2.0)),
    );
    let radicand = Expression::Binary(
        BinaryOperator::Add,
        Box::new(Expression::Number(-1.0)),
        Box::new(Expression::Binary(
            BinaryOperator::Multiply,
            Box::new(Expression::Number(
                std::f64::consts::LOG10_E * std::f64::consts::LOG10_E,
            )),
            Box::new(log_squared),
        )),
    );
    let numerator = Expression::Builtin(Builtin::Sqrt, vec![radicand]);
    if outer == Builtin::Sinh {
        return numerator;
    }
    let denominator = Expression::Builtin(
        Builtin::Cosh,
        vec![Expression::Builtin(
            Builtin::Acosh,
            vec![Expression::Builtin(Builtin::Log10, vec![argument.clone()])],
        )],
    );
    Expression::Binary(
        BinaryOperator::Divide,
        Box::new(numerator),
        Box::new(denominator),
    )
}

fn rewrite_tanh_acosh(argument: &Expression) -> Expression {
    let numerator = Expression::Builtin(
        Builtin::Sqrt,
        vec![Expression::Binary(
            BinaryOperator::Subtract,
            Box::new(Expression::Binary(
                BinaryOperator::Power,
                Box::new(argument.clone()),
                Box::new(Expression::Number(2.0)),
            )),
            Box::new(Expression::Number(1.0)),
        )],
    );
    let denominator = Expression::Builtin(
        Builtin::Cosh,
        vec![Expression::Builtin(Builtin::Acosh, vec![argument.clone()])],
    );
    Expression::Binary(
        BinaryOperator::Divide,
        Box::new(numerator),
        Box::new(denominator),
    )
}

fn rewrite_tanh_acosh_cotangent(argument: &Expression) -> Expression {
    Expression::Builtin(
        Builtin::If,
        vec![
            Expression::Binary(
                BinaryOperator::Less,
                Box::new(argument.clone()),
                Box::new(Expression::Number(1.0)),
            ),
            Expression::Builtin(Builtin::Acosh, vec![argument.clone()]),
            rewrite_tanh_acosh(argument),
        ],
    )
}

fn rewrite_tanh_asinh(argument: Expression) -> Expression {
    Expression::Binary(
        BinaryOperator::Divide,
        Box::new(argument.clone()),
        Box::new(Expression::Builtin(
            Builtin::Sqrt,
            vec![Expression::Binary(
                BinaryOperator::Add,
                Box::new(Expression::Binary(
                    BinaryOperator::Power,
                    Box::new(argument),
                    Box::new(Expression::Number(2.0)),
                )),
                Box::new(Expression::Number(1.0)),
            )],
        )),
    )
}

fn evaluate_binary(
    operator: BinaryOperator,
    left: f64,
    right: f64,
    epsilon: f64,
) -> Result<f64, EvalError> {
    Ok(match operator {
        BinaryOperator::Add => left + right,
        BinaryOperator::Subtract => left - right,
        BinaryOperator::Multiply => left * right,
        BinaryOperator::Divide => {
            if right == 0.0 {
                return Err(EvalError::DivisionByZero);
            }
            left / right
        }
        BinaryOperator::Modulo => {
            if right == 0.0 {
                return Err(EvalError::DivisionByZero);
            }
            left % right
        }
        BinaryOperator::Power => {
            if left == 0.0 && right < 0.0 {
                return Err(EvalError::InvalidLogarithmOrPower);
            }
            evaluate_power(left, right, epsilon)
        }
        BinaryOperator::Equal => f64::from((left - right).abs() <= epsilon),
        BinaryOperator::NotEqual => f64::from((left - right).abs() > epsilon),
        BinaryOperator::Less => f64::from(left < right - epsilon),
        BinaryOperator::LessOrEqual => f64::from(left <= right + epsilon),
        BinaryOperator::Greater => f64::from(right < left - epsilon),
        BinaryOperator::GreaterOrEqual => f64::from(right <= left + epsilon),
        BinaryOperator::And => f64::from(truth(left) && truth(right)),
        BinaryOperator::Or => f64::from(truth(left) || truth(right)),
    })
}

fn evaluate_power(base: f64, exponent: f64, epsilon: f64) -> f64 {
    if base == 1.0 {
        return 1.0;
    }
    if let Some(integer_exponent) = legacy_long_integer(exponent) {
        let magnitude = integer_exponent.unsigned_abs();
        let value = integer_power(base, magnitude);
        return if integer_exponent < 0 {
            1.0 / value
        } else {
            value
        };
    }
    if exponent >= 0.0 {
        if base > 0.0 {
            return (base.ln() * exponent).exp();
        }
        if base == 0.0 {
            return 0.0;
        }
        if !is_integer_with_epsilon(exponent * 16.0, epsilon) {
            return -((-base).ln() * exponent).exp();
        }
    } else {
        if base > 0.0 {
            return ((1.0 / base).ln() * -exponent).exp();
        }
        if base < 0.0 && !is_integer_with_epsilon(exponent * -16.0, epsilon) {
            return -((-1.0 / base).ln() * -exponent).exp();
        }
    }
    base.powf(exponent)
}

#[cfg(any(windows, target_pointer_width = "32"))]
fn legacy_long_integer(value: f64) -> Option<i64> {
    if value.fract() == 0.0 && value >= i32::MIN as f64 && value <= i32::MAX as f64 {
        Some(value as i32 as i64)
    } else {
        None
    }
}

#[cfg(all(not(windows), target_pointer_width = "64"))]
fn legacy_long_integer(value: f64) -> Option<i64> {
    if value.fract() == 0.0 && value > i64::MIN as f64 && value < -(i64::MIN as f64) {
        Some(value as i64)
    } else {
        None
    }
}

fn integer_power(mut base: f64, mut exponent: u64) -> f64 {
    let mut result = 1.0;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result *= base;
            exponent -= 1;
        } else {
            base *= base;
            exponent /= 2;
        }
    }
    result
}

fn is_integer_with_epsilon(value: f64, epsilon: f64) -> bool {
    (value - value.floor()).abs() <= epsilon
}

fn evaluate_builtin(
    builtin: Builtin,
    values: &[f64],
    use_degrees: bool,
    epsilon: f64,
    validate_errors: bool,
) -> Result<f64, EvalError> {
    let mut first = values[0];
    if use_degrees && builtin.angle_input() {
        first = first.to_radians();
    }
    let value = match builtin {
        Builtin::Abs => first.abs(),
        Builtin::Acos => {
            if validate_errors && (first < -1.0 || first > 1.0) {
                return Err(EvalError::InvalidInverseTrigonometricInput);
            }
            first.acos()
        }
        Builtin::Acosh => {
            if validate_errors && first < 1.0 {
                return Err(EvalError::InvalidInverseTrigonometricInput);
            }
            (first + (first * first - 1.0).sqrt()).ln()
        }
        Builtin::Asin => {
            if validate_errors && (first < -1.0 || first > 1.0) {
                return Err(EvalError::InvalidInverseTrigonometricInput);
            }
            first.asin()
        }
        Builtin::Asinh => (first + (first * first + 1.0).sqrt()).ln(),
        Builtin::Atan => first.atan(),
        Builtin::Atan2 => values[0].atan2(values[1]),
        Builtin::Atanh => {
            if validate_errors && (first <= -1.0 || first >= 1.0) {
                return Err(EvalError::InvalidInverseTrigonometricInput);
            }
            0.5 * ((1.0 + first) / (1.0 - first)).ln()
        }
        Builtin::Cbrt => {
            if first > 0.0 {
                (first.ln() / 3.0).exp()
            } else if first < 0.0 {
                -((-first).ln() / 3.0).exp()
            } else {
                0.0
            }
        }
        Builtin::Ceil => first.ceil(),
        Builtin::Cos => first.cos(),
        Builtin::Cosh => first.cosh(),
        Builtin::Cot => {
            let tangent = first.tan();
            if validate_errors && tangent == 0.0 {
                return Err(EvalError::DivisionByZero);
            }
            1.0 / tangent
        }
        Builtin::Csc => {
            let sine = first.sin();
            if validate_errors && sine == 0.0 {
                return Err(EvalError::DivisionByZero);
            }
            1.0 / sine
        }
        Builtin::Exp => first.exp(),
        Builtin::Exp2 => evaluate_power(2.0, first, epsilon),
        Builtin::Floor => first.floor(),
        Builtin::Hypot => (values[0] * values[0] + values[1] * values[1]).sqrt(),
        Builtin::If => unreachable!("conditional evaluation is handled before arguments"),
        Builtin::Int => {
            if first < 0.0 {
                (first - 0.5).ceil()
            } else {
                (first + 0.5).floor()
            }
        }
        Builtin::Log | Builtin::Log10 | Builtin::Log2 => {
            if validate_errors && (first <= 0.0 || first.is_nan()) {
                return Err(EvalError::InvalidLogarithmOrPower);
            }
            match builtin {
                Builtin::Log => first.ln(),
                Builtin::Log10 => first.ln() * std::f64::consts::LOG10_E,
                Builtin::Log2 => first.ln() * std::f64::consts::LOG2_E,
                _ => unreachable!(),
            }
        }
        Builtin::Max => {
            if values[0] > values[1] {
                values[0]
            } else {
                values[1]
            }
        }
        Builtin::Min => {
            if values[0] < values[1] {
                values[0]
            } else {
                values[1]
            }
        }
        Builtin::Pow => {
            if validate_errors && values[0] == 0.0 && values[1] < 0.0 {
                return Err(EvalError::InvalidLogarithmOrPower);
            }
            evaluate_power(values[0], values[1], epsilon)
        }
        Builtin::Sec => {
            let cosine = first.cos();
            if validate_errors && cosine == 0.0 {
                return Err(EvalError::DivisionByZero);
            }
            1.0 / cosine
        }
        Builtin::Sin => first.sin(),
        Builtin::Sinh => first.sinh(),
        Builtin::Sqrt => {
            if validate_errors && first < 0.0 {
                return Err(EvalError::NegativeSquareRoot);
            }
            first.sqrt()
        }
        Builtin::Tan => first.tan(),
        Builtin::Tanh => first.tanh(),
        Builtin::Trunc => first.trunc(),
    };
    Ok(if use_degrees && builtin.angle_output() {
        value.to_degrees()
    } else {
        value
    })
}

fn fold_constants(expression: Expression, use_degrees: bool, epsilon: f64) -> Expression {
    match expression {
        Expression::Shared(index, child) => Expression::Shared(
            index,
            Box::new(fold_constants(*child, use_degrees, epsilon)),
        ),
        Expression::Unary(operator, child) => {
            let child = fold_constants(*child, use_degrees, epsilon);
            if let Expression::Number(value) = child {
                let folded = match operator {
                    UnaryOperator::Negate => -value,
                    UnaryOperator::Not => f64::from(!truth(value)),
                    UnaryOperator::ExactZero => {
                        if value == 0.0 {
                            f64::INFINITY
                        } else {
                            0.0
                        }
                    }
                    UnaryOperator::ExactNegative => {
                        if value < 0.0 {
                            f64::INFINITY
                        } else {
                            0.0
                        }
                    }
                };
                Expression::Number(folded)
            } else {
                Expression::Unary(operator, Box::new(child))
            }
        }
        Expression::Binary(operator, left, right) => {
            let left = fold_constants(*left, use_degrees, epsilon);
            let right = fold_constants(*right, use_degrees, epsilon);
            let left_number = match &left {
                Expression::Number(value) => Some(*value),
                _ => None,
            };
            let right_number = match &right {
                Expression::Number(value) => Some(*value),
                _ => None,
            };
            match operator {
                BinaryOperator::Add if is_trigonometric_square_sum(&left, &right) => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::Subtract if same_expression(&left, &right) => {
                    return Expression::Number(0.0);
                }
                BinaryOperator::Equal if same_expression(&left, &right) => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::NotEqual if same_expression(&left, &right) => {
                    return Expression::Number(0.0);
                }
                BinaryOperator::Less if same_expression(&left, &right) => {
                    return Expression::Number(0.0);
                }
                BinaryOperator::LessOrEqual if same_expression(&left, &right) => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::Greater if same_expression(&left, &right) => {
                    return Expression::Number(0.0);
                }
                BinaryOperator::GreaterOrEqual if same_expression(&left, &right) => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::Divide if left_number == Some(0.0) && right_number == Some(0.0) => {
                    return Expression::Number(0.0);
                }
                BinaryOperator::Divide if right_number == Some(0.0) => {
                    return Expression::Number(0.0);
                }
                BinaryOperator::Divide if left_number == Some(0.0) && right_number.is_none() => {
                    if let Some(value) = left_number {
                        return Expression::Number(value);
                    }
                }
                BinaryOperator::Divide if same_expression(&left, &right) => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::Add if right_number == Some(0.0) && left_number.is_none() => {
                    return left;
                }
                BinaryOperator::Add if left_number == Some(0.0) && right_number.is_none() => {
                    return right;
                }
                BinaryOperator::Multiply if left_number == Some(1.0) && right_number.is_none() => {
                    return right;
                }
                BinaryOperator::Multiply if right_number == Some(1.0) && left_number.is_none() => {
                    return left;
                }
                BinaryOperator::Subtract if left_number == Some(0.0) && right_number.is_none() => {
                    return Expression::Unary(UnaryOperator::Negate, Box::new(right));
                }
                BinaryOperator::Divide if right_number == Some(1.0) && left_number.is_none() => {
                    return left;
                }
                BinaryOperator::Power if right_number == Some(1.0) && left_number.is_none() => {
                    return left;
                }
                BinaryOperator::Power if right_number == Some(0.0) && left_number.is_none() => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::Power if left_number == Some(1.0) && right_number.is_none() => {
                    return Expression::Number(1.0);
                }
                BinaryOperator::Power
                    if left_number == Some(0.0)
                        && right_number.is_some_and(|value| value < 0.0) =>
                {
                    return Expression::Number(0.0);
                }
                BinaryOperator::Multiply
                    if (left_number == Some(0.0) || right_number == Some(0.0))
                        && (!(left_number.is_some() && right_number.is_some())
                            || left_number.is_some_and(|value| !value.is_finite())
                            || right_number.is_some_and(|value| !value.is_finite())) =>
                {
                    if let Some(value) = left_number.filter(|value| *value == 0.0) {
                        return Expression::Number(value);
                    }
                    if let Some(value) = right_number.filter(|value| *value == 0.0) {
                        return Expression::Number(value);
                    }
                }
                BinaryOperator::And
                    if left_number.is_some_and(|value| !truth(value))
                        || right_number.is_some_and(|value| !truth(value)) =>
                {
                    return Expression::Number(0.0);
                }
                BinaryOperator::Or
                    if left_number.is_some_and(truth) || right_number.is_some_and(truth) =>
                {
                    return Expression::Number(1.0);
                }
                _ => {}
            }
            if let (Expression::Number(left_value), Expression::Number(right_value)) =
                (&left, &right)
            {
                if let Ok(value) = evaluate_binary(operator, *left_value, *right_value, epsilon) {
                    return Expression::Number(value);
                }
            }
            Expression::Binary(operator, Box::new(left), Box::new(right))
        }
        Expression::Builtin(builtin, arguments) => {
            if matches!(builtin, Builtin::Sec | Builtin::Tan) {
                if let [Expression::Builtin(Builtin::Asin, asin_arguments)] = arguments.as_slice() {
                    if let [Expression::Builtin(Builtin::Cosh, cosh_arguments)] =
                        asin_arguments.as_slice()
                    {
                        if matches!(cosh_arguments.as_slice(), [Expression::Variable(_)]) {
                            let angle = if use_degrees {
                                90.0
                            } else {
                                std::f64::consts::FRAC_PI_2
                            };
                            return Expression::Binary(
                                BinaryOperator::Multiply,
                                Box::new(Expression::Number(0.5)),
                                Box::new(Expression::Builtin(
                                    builtin,
                                    vec![Expression::Number(angle)],
                                )),
                            );
                        }
                    }
                }
            }
            let arguments = arguments
                .into_iter()
                .map(|argument| fold_constants(argument, use_degrees, epsilon))
                .collect::<Vec<_>>();
            if let [Expression::Builtin(exponential, inner_arguments)] = arguments.as_slice() {
                if let [argument] = inner_arguments.as_slice() {
                    if let Some(rewritten) = rewrite_exp_log(builtin, *exponential, argument) {
                        return rewritten;
                    }
                    if let Some(rewritten) = rewrite_sixth_root(builtin, *exponential, argument) {
                        return rewritten;
                    }
                    if let Some(rewritten) = rewrite_log_exp(builtin, *exponential, argument) {
                        return rewritten;
                    }
                    if builtin == Builtin::Sinh && *exponential == Builtin::Acosh {
                        return rewrite_sinh_acosh(argument.clone());
                    }
                    if builtin == Builtin::Tanh && *exponential == Builtin::Acosh {
                        if matches!(argument, Expression::Builtin(Builtin::Cot, _)) {
                            return rewrite_tanh_acosh_cotangent(argument);
                        }
                        return rewrite_tanh_acosh(argument);
                    }
                    if builtin == Builtin::Tanh && *exponential == Builtin::Asinh {
                        return rewrite_tanh_asinh(argument.clone());
                    }
                    if builtin == Builtin::Tanh
                        && *exponential == Builtin::Abs
                        && !matches!(
                            argument,
                            Expression::Builtin(Builtin::Cosh | Builtin::Exp | Builtin::Exp2, _)
                        )
                    {
                        return Expression::Binary(
                            BinaryOperator::Divide,
                            Box::new(Expression::Builtin(
                                Builtin::Sinh,
                                vec![Expression::Builtin(Builtin::Abs, vec![argument.clone()])],
                            )),
                            Box::new(Expression::Builtin(Builtin::Cosh, vec![argument.clone()])),
                        );
                    }
                    if let Some(rewritten) = rewrite_tan_cot_acos(builtin, *exponential, argument) {
                        return rewritten;
                    }
                    if builtin == Builtin::Sqrt {
                        if let Some(rewritten) = rewrite_sqrt_reciprocal(*exponential, argument) {
                            return rewritten;
                        }
                    }
                }
            }
            if builtin == Builtin::Abs {
                if let [Expression::Builtin(Builtin::Cbrt, cbrt_arguments)] = arguments.as_slice() {
                    if !matches!(
                        cbrt_arguments.as_slice(),
                        [Expression::Builtin(
                            Builtin::Asin
                                | Builtin::Atan
                                | Builtin::Cos
                                | Builtin::Cot
                                | Builtin::Csc
                                | Builtin::Sec
                                | Builtin::Sin
                                | Builtin::Sqrt,
                            _
                        )]
                    ) {
                        return arguments
                            .into_iter()
                            .next()
                            .expect("one argument was matched");
                    }
                }
                if let [Expression::Builtin(Builtin::Asin, inner_arguments)] = arguments.as_slice()
                {
                    if matches!(
                        inner_arguments.as_slice(),
                        [Expression::Builtin(Builtin::Cbrt | Builtin::Sqrt, _)]
                    ) {
                        return Expression::Builtin(Builtin::Asin, inner_arguments.clone());
                    }
                }
                if let [Expression::Builtin(Builtin::Tanh, inner_arguments)] = arguments.as_slice()
                {
                    if let [argument] = inner_arguments.as_slice() {
                        // The sinh/cosh ratio overflows to inf/inf for these growing inputs.
                        if !matches!(
                            argument,
                            Expression::Builtin(Builtin::Cosh | Builtin::Exp | Builtin::Exp2, _)
                        ) {
                            return Expression::Binary(
                                BinaryOperator::Divide,
                                Box::new(Expression::Builtin(
                                    Builtin::Abs,
                                    vec![Expression::Builtin(
                                        Builtin::Sinh,
                                        vec![argument.clone()],
                                    )],
                                )),
                                Box::new(Expression::Builtin(
                                    Builtin::Cosh,
                                    vec![argument.clone()],
                                )),
                            );
                        }
                    }
                }
            }
            if matches!(builtin, Builtin::Acos | Builtin::Asin) {
                if let [Expression::Builtin(Builtin::Exp | Builtin::Exp2, exponential_arguments)] =
                    arguments.as_slice()
                {
                    if matches!(
                        exponential_arguments.as_slice(),
                        [Expression::Builtin(
                            Builtin::Acos | Builtin::Exp | Builtin::Exp2,
                            _
                        )]
                    ) {
                        return Expression::Number(if builtin == Builtin::Acos {
                            0.0
                        } else {
                            std::f64::consts::FRAC_PI_2
                        });
                    }
                }
                if let [Expression::Builtin(Builtin::Cosh, inner_arguments)] = arguments.as_slice()
                {
                    if !matches!(
                        inner_arguments.as_slice(),
                        [Expression::Builtin(Builtin::Cosh, _)]
                    ) {
                        return Expression::Number(if builtin == Builtin::Acos {
                            0.0
                        } else {
                            std::f64::consts::FRAC_PI_2
                        });
                    }
                }
            }
            if let [Expression::Builtin(inner, inner_arguments)] = arguments.as_slice() {
                let reciprocal_identity = matches!(
                    (builtin, inner),
                    (Builtin::Sec, Builtin::Acos) | (Builtin::Csc, Builtin::Asin)
                );
                if reciprocal_identity {
                    if let [argument] = inner_arguments.as_slice() {
                        return Expression::Binary(
                            BinaryOperator::Divide,
                            Box::new(Expression::Number(1.0)),
                            Box::new(argument.clone()),
                        );
                    }
                }
            }
            match (builtin, arguments.as_slice()) {
                (Builtin::Sin, [Expression::Builtin(Builtin::Asin, inner)])
                | (Builtin::Cos, [Expression::Builtin(Builtin::Acos, inner)])
                | (Builtin::Sinh, [Expression::Builtin(Builtin::Asinh, inner)])
                | (Builtin::Exp, [Expression::Builtin(Builtin::Log, inner)])
                | (Builtin::Exp2, [Expression::Builtin(Builtin::Log2, inner)]) => {
                    return inner[0].clone();
                }
                (Builtin::Atan, [Expression::Builtin(Builtin::Tan, inner)]) => {
                    if let [argument] = inner.as_slice() {
                        return Expression::Builtin(
                            Builtin::Atan2,
                            vec![
                                Expression::Builtin(Builtin::Sin, vec![argument.clone()]),
                                Expression::Builtin(Builtin::Cos, vec![argument.clone()]),
                            ],
                        );
                    }
                }
                _ => {}
            }
            if builtin == Builtin::Pow {
                match arguments.as_slice() {
                    [Expression::Number(1.0), _] => return Expression::Number(1.0),
                    [_, Expression::Number(0.0)] => return Expression::Number(1.0),
                    [Expression::Number(base), Expression::Number(exponent)]
                        if *base == 0.0 && *exponent < 0.0 =>
                    {
                        return Expression::Number(0.0);
                    }
                    _ => {}
                }
            }
            if arguments
                .iter()
                .all(|argument| matches!(argument, Expression::Number(_)))
            {
                if builtin == Builtin::If {
                    if let Expression::Number(condition) = arguments[0] {
                        return arguments[if truth(condition) { 1 } else { 2 }].clone();
                    }
                } else {
                    let values = arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::Number(value) => *value,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    if let Ok(value) =
                        evaluate_builtin(builtin, &values, use_degrees, epsilon, false)
                    {
                        return Expression::Number(value);
                    }
                }
            }
            Expression::Builtin(builtin, arguments)
        }
        Expression::UserFunction(function, arguments) => Expression::UserFunction(
            function,
            arguments
                .into_iter()
                .map(|argument| fold_constants(argument, use_degrees, epsilon))
                .collect(),
        ),
        other => other,
    }
}

const LEGACY_OPTIMIZER_LOG10_E: f64 = 0.434_294_481_903_251_87;
const LEGACY_OPTIMIZER_LOG10_2: f64 = 0.301_029_995_663_981_25;
const LEGACY_LOG2_E_LN: f64 = 0.366_512_920_581_664_3;
const LEGACY_SQRT_EXP2_FACTOR: f64 = 0.346_573_590_279_972_6;

fn rewrite_legacy_log_of_log2(builtin: Builtin, argument: &Expression) -> Option<Expression> {
    if !matches!(builtin, Builtin::Log | Builtin::Log10 | Builtin::Log2) {
        return None;
    };
    let Expression::Builtin(Builtin::Log2, inner) = argument else {
        return None;
    };
    let [inner] = inner.as_slice() else {
        return None;
    };
    let constant = match builtin {
        Builtin::Log => LEGACY_LOG2_E_LN,
        Builtin::Log10 => LEGACY_LOG2_E_LN * std::f64::consts::LOG10_E,
        Builtin::Log2 => LEGACY_LOG2_E_LN * std::f64::consts::LOG2_E,
        _ => unreachable!(),
    };
    Some(Expression::Binary(
        BinaryOperator::Add,
        Box::new(Expression::Number(constant)),
        Box::new(Expression::Builtin(
            builtin,
            vec![Expression::Builtin(Builtin::Log, vec![inner.clone()])],
        )),
    ))
}

fn rewrite_legacy_integer_log_of_log2(argument: &Expression) -> Option<Expression> {
    let Expression::Builtin(logarithm @ (Builtin::Log | Builtin::Log10), arguments) = argument
    else {
        return None;
    };
    let [Expression::Builtin(Builtin::Log2, log2_arguments)] = arguments.as_slice() else {
        return None;
    };
    let [inner] = log2_arguments.as_slice() else {
        return None;
    };
    let (logarithm, offset) = match logarithm {
        Builtin::Log => (Builtin::Log, 0.866_512_920_581_664_3),
        Builtin::Log10 => (Builtin::Log10, 0.659_174_538_954_861_6),
        _ => unreachable!(),
    };
    Some(Expression::Builtin(
        Builtin::Floor,
        vec![Expression::Binary(
            BinaryOperator::Add,
            Box::new(Expression::Builtin(
                logarithm,
                vec![Expression::Builtin(Builtin::Log, vec![inner.clone()])],
            )),
            Box::new(Expression::Number(offset)),
        )],
    ))
}

fn rewrite_legacy_root_of_nested_log2(root: Builtin, argument: &Expression) -> Option<Expression> {
    let Expression::Builtin(Builtin::Log2, arguments) = argument else {
        return None;
    };
    let [Expression::Builtin(Builtin::Log2, nested_arguments)] = arguments.as_slice() else {
        return None;
    };
    let [inner] = nested_arguments.as_slice() else {
        return None;
    };
    let factor = match root {
        Builtin::Cbrt => 1.1299472763373901,
        Builtin::Sqrt => 1.2011224087864498,
        _ => return None,
    };
    Some(Expression::Binary(
        BinaryOperator::Multiply,
        Box::new(Expression::Number(factor)),
        Box::new(Expression::Builtin(
            root,
            vec![Expression::Binary(
                BinaryOperator::Add,
                Box::new(Expression::Number(LEGACY_LOG2_E_LN)),
                Box::new(Expression::Builtin(
                    Builtin::Log,
                    vec![Expression::Builtin(Builtin::Log, vec![inner.clone()])],
                )),
            )],
        )),
    ))
}

fn rewrite_legacy_math_lowering(expression: Expression) -> Expression {
    match expression {
        Expression::Number(_) | Expression::Variable(_) | Expression::Inline(_) => expression,
        Expression::Shared(slot, expression) => {
            Expression::Shared(slot, Box::new(rewrite_legacy_math_lowering(*expression)))
        }
        Expression::Unary(operator, expression) => Expression::Unary(
            operator,
            Box::new(rewrite_legacy_math_lowering(*expression)),
        ),
        Expression::Binary(operator, left, right) => {
            let left = rewrite_legacy_math_lowering(*left);
            let right = rewrite_legacy_math_lowering(*right);
            if operator == BinaryOperator::Power && matches!(right, Expression::Number(2.0)) {
                if let Expression::Builtin(Builtin::Log2, inner) = &left {
                    if let [argument] = inner.as_slice() {
                        return Expression::Binary(
                            BinaryOperator::Multiply,
                            Box::new(Expression::Number(
                                std::f64::consts::LOG2_E * std::f64::consts::LOG2_E,
                            )),
                            Box::new(Expression::Binary(
                                BinaryOperator::Power,
                                Box::new(Expression::Builtin(Builtin::Log, vec![argument.clone()])),
                                Box::new(Expression::Number(2.0)),
                            )),
                        );
                    }
                }
            }
            Expression::Binary(operator, Box::new(left), Box::new(right))
        }
        Expression::Builtin(builtin, arguments) => {
            if let [argument] = arguments.as_slice() {
                if builtin == Builtin::Int {
                    if let Some(rewritten) = rewrite_legacy_integer_log_of_log2(argument) {
                        return rewritten;
                    }
                }
                if let Some(rewritten) = rewrite_legacy_root_of_nested_log2(builtin, argument) {
                    return rewritten;
                }
            }
            // Preserve the legacy finite-precision tangent result inside abs.
            let preserve_tan_atan = builtin == Builtin::Abs
                && matches!(
                    arguments.as_slice(),
                    [Expression::Builtin(Builtin::Tan, tan_arguments)]
                        if matches!(
                            tan_arguments.as_slice(),
                            [Expression::Builtin(Builtin::Atan, _)]
                        )
                );
            let arguments = if preserve_tan_atan {
                arguments
            } else {
                arguments
                    .into_iter()
                    .map(rewrite_legacy_math_lowering)
                    .collect::<Vec<_>>()
            };
            match (builtin, arguments.as_slice()) {
                (Builtin::Exp2, [argument]) if !matches!(argument, Expression::Number(_)) => {
                    Expression::Builtin(
                        Builtin::Exp,
                        vec![Expression::Binary(
                            BinaryOperator::Multiply,
                            Box::new(Expression::Number(std::f64::consts::LN_2)),
                            Box::new(argument.clone()),
                        )],
                    )
                }
                (Builtin::Tan, [Expression::Builtin(Builtin::Atan, inner)]) => inner[0].clone(),
                (Builtin::Cot, [Expression::Builtin(Builtin::Atan, inner)]) => {
                    if let [argument] = inner.as_slice() {
                        Expression::Binary(
                            BinaryOperator::Divide,
                            Box::new(Expression::Number(1.0)),
                            Box::new(argument.clone()),
                        )
                    } else {
                        Expression::Builtin(builtin, arguments)
                    }
                }
                (Builtin::Log | Builtin::Log10 | Builtin::Log2, [argument]) => {
                    rewrite_legacy_log_of_log2(builtin, argument)
                        .unwrap_or_else(|| Expression::Builtin(builtin, arguments))
                }
                (Builtin::Sqrt, [Expression::Builtin(Builtin::Exp, inner)]) => {
                    if let [Expression::Binary(BinaryOperator::Multiply, factor, argument)] =
                        inner.as_slice()
                    {
                        if let Expression::Number(value) = &**factor {
                            let value = if *value == std::f64::consts::LN_2 {
                                LEGACY_SQRT_EXP2_FACTOR
                            } else {
                                value * 0.5
                            };
                            return Expression::Builtin(
                                Builtin::Exp,
                                vec![Expression::Binary(
                                    BinaryOperator::Multiply,
                                    Box::new(Expression::Number(value)),
                                    argument.clone(),
                                )],
                            );
                        }
                    }
                    Expression::Builtin(builtin, arguments)
                }
                (
                    Builtin::Sqrt,
                    [Expression::Binary(BinaryOperator::Multiply, factor, argument)],
                ) => {
                    if let (Expression::Number(factor), Expression::Builtin(Builtin::Log, inner)) =
                        (&**factor, &**argument)
                    {
                        if *factor > 0.0 {
                            return Expression::Binary(
                                BinaryOperator::Multiply,
                                Box::new(Expression::Number(factor.sqrt())),
                                Box::new(Expression::Builtin(
                                    Builtin::Sqrt,
                                    vec![Expression::Builtin(Builtin::Log, inner.clone())],
                                )),
                            );
                        }
                    }
                    Expression::Builtin(builtin, arguments)
                }
                (Builtin::Sqrt, [Expression::Builtin(Builtin::Log2, inner)]) => {
                    if let [argument] = inner.as_slice() {
                        return Expression::Binary(
                            BinaryOperator::Multiply,
                            Box::new(Expression::Number(std::f64::consts::LOG2_E.sqrt())),
                            Box::new(Expression::Builtin(
                                Builtin::Sqrt,
                                vec![Expression::Builtin(Builtin::Log, vec![argument.clone()])],
                            )),
                        );
                    }
                    Expression::Builtin(builtin, arguments)
                }
                _ => Expression::Builtin(builtin, arguments),
            }
        }
        Expression::UserFunction(function, arguments) => Expression::UserFunction(
            function,
            arguments
                .into_iter()
                .map(rewrite_legacy_math_lowering)
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn parse(expression: &str, variables: &str, degrees: bool) -> Parser {
        let mut parser = Parser::new();
        parser.parse(expression, variables, degrees).unwrap();
        parser
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn precedence_associativity_and_unary_power_match_reference() {
        close(parse("-2^2", "", false).eval(&[]).unwrap(), -4.0);
        close(parse("2^3^2", "", false).eval(&[]).unwrap(), 512.0);
        close(parse("2^-2", "", false).eval(&[]).unwrap(), 0.25);
        close(parse("2+3*4", "", false).eval(&[]).unwrap(), 14.0);
    }

    #[test]
    fn deeply_nested_expressions_are_rejected_without_overflowing() {
        let deep = format!("{}1{}", "(".repeat(4000), ")".repeat(4000));
        let mut parser = Parser::new();
        let error = parser.parse(&deep, "", false).expect_err("nesting limit");
        assert!(matches!(error.kind, ParseErrorKind::OutOfMemory));

        // A reasonable depth still parses.
        let reasonable = format!("{}1{}", "(".repeat(100), ")".repeat(100));
        let mut parser = Parser::new();
        parser.parse(&reasonable, "", false).expect("100 levels");
        close(parser.eval(&[]).unwrap(), 1.0);
    }

    #[test]
    fn units_degrees_and_builtin_functions_evaluate() {
        let mut parser = Parser::new();
        assert!(parser.add_constant("pi", std::f64::consts::PI));
        assert!(parser.add_unit("cm", 0.01));
        parser.parse("sin(30)+2cm", "", true).unwrap();
        close(parser.eval(&[]).unwrap(), 0.52);

        let mut parser = parse("asin(0.5)", "", true);
        close(parser.eval(&[]).unwrap(), 30.0);
        parser.parse("int(-1.5)+trunc(-1.5)", "", false).unwrap();
        close(parser.eval(&[]).unwrap(), -3.0);
    }

    #[test]
    fn registered_identifier_names_cannot_collide_or_be_replaced() {
        let mut parser = Parser::new();
        assert!(parser.add_constant("scale", 2.0));
        assert!(!parser.add_constant("scale", 3.0));
        assert!(!parser.add_unit("scale", 0.01));
        assert!(!parser.add_function("scale", 1, |_| 0.0));

        assert!(parser.add_unit("cm", 0.01));
        assert!(!parser.add_constant("cm", 0.02));
        assert!(!parser.add_unit("cm", 0.02));
        assert!(!parser.add_function("cm", 1, |_| 0.0));

        assert!(parser.add_function("custom", 1, |_| 0.0));
        assert!(!parser.add_function("custom", 1, |_| 1.0));
        assert!(!parser.add_constant("custom", 1.0));
        assert!(!parser.add_unit("custom", 1.0));
        assert!(!parser.add_constant("sin", 0.0));
    }

    #[test]
    fn removing_identifiers_affects_future_parses_not_compiled_expressions() {
        let mut parser = Parser::new();
        assert!(parser.add_constant("scale", 2.0));
        parser.parse("scale*x", "x", false).unwrap();
        assert!(parser.remove_identifier("scale"));
        assert_eq!(parser.eval(&[3.0]).unwrap(), 6.0);
        assert_eq!(
            parser.parse("scale*x", "x", false).unwrap_err().kind,
            ParseErrorKind::UnknownIdentifier
        );
        assert!(!parser.remove_identifier("x"));
        assert!(!parser.remove_identifier("scale"));
    }

    #[test]
    fn cloned_parsers_keep_independent_programs_and_identifiers() {
        let mut original = Parser::new();
        assert!(original.add_constant("scale", 2.0));
        original.parse("scale*x", "x", false).unwrap();

        let mut cloned = original.clone();
        assert!(cloned.remove_identifier("scale"));
        cloned.parse("3*x", "x", false).unwrap();

        assert_eq!(original.eval(&[4.0]).unwrap(), 8.0);
        assert_eq!(cloned.eval(&[4.0]).unwrap(), 12.0);
        assert!(original.parse("scale*x", "x", false).is_ok());
        assert_eq!(
            cloned.parse("scale*x", "x", false).unwrap_err().kind,
            ParseErrorKind::UnknownIdentifier
        );
    }

    #[test]
    fn identifier_rules_match_the_original_utf8_scanner() {
        assert!(is_identifier("🧪value"));
        assert!(is_identifier(" value"));
        assert!(!is_identifier("\u{00a0}value"));
        assert!(!is_identifier("value\u{200b}tail"));
        assert!(tokenize("x\u{200b}+y").is_ok());
    }

    #[test]
    fn identifier_character_rules_cover_all_unicode_scalars() {
        for codepoint in 0..=0x10ffff {
            let Some(character) = char::from_u32(codepoint) else {
                continue;
            };
            let legacy_space = matches!(
                codepoint,
                0x09..=0x0b
                    | 0x0d
                    | 0x20
                    | 0x00a0
                    | 0x2000..=0x200b
                    | 0x202f
                    | 0x205f
                    | 0x3000
            );
            let ascii_letter = matches!(codepoint, 0x41..=0x5a | 0x61..=0x7a);
            let expected_first = if character.is_ascii() {
                ascii_letter || codepoint == 0x5f
            } else {
                !legacy_space
            };
            let expected_continuation = if character.is_ascii() {
                ascii_letter || matches!(codepoint, 0x30..=0x39 | 0x5f)
            } else {
                !legacy_space
            };
            assert_eq!(
                is_identifier_character(character, true),
                expected_first,
                "U+{codepoint:04X} identifier start"
            );
            assert_eq!(
                is_identifier_character(character, false),
                expected_continuation,
                "U+{codepoint:04X} identifier continuation"
            );
        }
    }

    #[test]
    fn every_default_real_builtin_parses_and_evaluates() {
        let mut parser = Parser::new();
        assert!(parser.add_constant("pi", std::f64::consts::PI));
        let cases = [
            ("abs(-3)", 3.0),
            ("acos(1)", 0.0),
            ("acosh(1)", 0.0),
            ("asin(0)", 0.0),
            ("asinh(0)", 0.0),
            ("atan(0)", 0.0),
            ("atan2(0,1)", 0.0),
            ("atanh(0)", 0.0),
            ("cbrt(-8)", -2.0),
            ("ceil(1.2)", 2.0),
            ("cos(0)", 1.0),
            ("cosh(0)", 1.0),
            ("cot(pi/4)", 1.0),
            ("csc(pi/2)", 1.0),
            ("exp(0)", 1.0),
            ("exp2(3)", 8.0),
            ("floor(1.8)", 1.0),
            ("hypot(3,4)", 5.0),
            ("if(0,1,2)", 2.0),
            ("int(-1.5)", -2.0),
            ("log(1)", 0.0),
            ("log10(100)", 2.0),
            ("log2(8)", 3.0),
            ("max(1,2)", 2.0),
            ("min(1,2)", 1.0),
            ("pow(2,3)", 8.0),
            ("sec(0)", 1.0),
            ("sin(pi/2)", 1.0),
            ("sinh(0)", 0.0),
            ("sqrt(4)", 2.0),
            ("tan(0)", 0.0),
            ("tanh(0)", 0.0),
            ("trunc(-1.8)", -1.0),
        ];
        for (expression, expected) in cases {
            parser.parse(expression, "", false).unwrap();
            close(parser.eval(&[]).unwrap(), expected);
        }
        parser.parse("cbrt(-0)", "", false).unwrap();
        assert_eq!(parser.eval(&[]).unwrap().to_bits(), 0.0_f64.to_bits());
        parser.parse("int(x)", "x", false).unwrap();
        assert_eq!(parser.eval(&[-0.0]).unwrap().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn epsilon_comparisons_and_truth_threshold_match_reference() {
        let previous_epsilon = Parser::epsilon();
        let default_parser = parse("x=y", "x,y", false);
        assert_eq!(default_parser.eval(&[1.0, 1.0 + 5.0e-13]).unwrap(), 1.0);
        assert_eq!(default_parser.eval(&[1.0, 1.0 + 2.0e-12]).unwrap(), 0.0);
        Parser::set_epsilon(1.0e-3);
        let mut parser = Parser::new();
        parser.parse("x=y", "x,y", false).unwrap();
        assert_eq!(parser.eval(&[1.0, 1.0005]).unwrap(), 1.0);
        assert_eq!(parser.eval(&[1.0, 1.002]).unwrap(), 0.0);
        assert_eq!(parse("!x", "x", false).eval(&[0.49]).unwrap(), 1.0);
        assert_eq!(parse("!!x", "x", false).eval(&[0.5]).unwrap(), 1.0);
        Parser::set_epsilon(previous_epsilon);
    }

    #[test]
    fn conditionals_are_lazy_but_logical_operators_are_eager() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_function = Arc::clone(&calls);
        let mut parser = Parser::new();
        assert!(parser.add_function("count", 1, move |values| {
            calls_for_function.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        parser.parse("if(0,count(x),7)", "x", false).unwrap();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 7.0);
        parser.parse("0&count(x)", "x", false).unwrap();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 0.0);
        parser.parse("1|count(x)", "x", false).unwrap();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 1.0);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn inline_variables_are_evaluated_once_and_in_order() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_function = Arc::clone(&calls);
        let mut parser = Parser::new();
        assert!(parser.add_function("count", 1, move |values| {
            calls_for_function.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        parser
            .parse("a:=count(x); b:=a+1; b+b", "x", false)
            .unwrap();
        assert_eq!(parser.eval(&[2.0]).unwrap(), 6.0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn evaluator_errors_keep_legacy_codes() {
        let cases = [
            ("1/0", EvalError::DivisionByZero, 1),
            ("sqrt(-1)", EvalError::NegativeSquareRoot, 2),
            ("log(0)", EvalError::InvalidLogarithmOrPower, 3),
            ("asin(2)", EvalError::InvalidInverseTrigonometricInput, 4),
        ];
        for (expression, expected, code) in cases {
            let error = parse(expression, "", false).eval(&[]).unwrap_err();
            assert_eq!(error, expected);
            assert_eq!(error.legacy_code(), Some(code));
        }
        assert_eq!(
            parse("log(x)", "x", false).eval(&[f64::NAN]),
            Err(EvalError::InvalidLogarithmOrPower)
        );
    }

    #[test]
    fn zero_negative_power_error_depends_on_compile_time_exponent() {
        for expression in ["0^-1", "pow(0,-1)", "0^-0.5"] {
            assert_eq!(
                parse(expression, "", false).eval(&[]).unwrap_err(),
                EvalError::DivisionByZero
            );
        }

        let mut parser = parse("x^y", "x,y", false);
        assert_eq!(
            parser.eval(&[0.0, -1.0]).unwrap_err(),
            EvalError::InvalidLogarithmOrPower
        );
        parser.parse("pow(x,y)", "x,y", false).unwrap();
        assert_eq!(
            parser.eval(&[0.0, -1.0]).unwrap_err(),
            EvalError::InvalidLogarithmOrPower
        );
        parser.parse("pow(x,-1)", "x", false).unwrap();
        assert_eq!(parser.eval(&[0.0]).unwrap_err(), EvalError::DivisionByZero);
    }

    #[test]
    fn fractional_power_of_negative_base_matches_legacy_real_semantics() {
        let parser = parse("x^y", "x,y", false);
        close(
            parser.eval(&[-2.0, 0.49]).unwrap(),
            -(2.0_f64.ln() * 0.49).exp(),
        );
        close(
            parser.eval(&[-2.0, -0.49]).unwrap(),
            -((0.5_f64).ln() * 0.49).exp(),
        );
        assert!(parser.eval(&[-2.0, 0.5]).unwrap().is_nan());
    }

    #[test]
    fn integer_power_uses_the_platform_c_long_range() {
        assert_eq!(integer_power(-2.0, 3), -8.0);
        assert_eq!(legacy_long_integer(i32::MAX as f64), Some(i32::MAX as i64));
        #[cfg(windows)]
        assert_eq!(legacy_long_integer(2_147_483_648.0), None);
        #[cfg(all(not(windows), target_pointer_width = "64"))]
        assert_eq!(legacy_long_integer(2_147_483_648.0), Some(2_147_483_648));
    }

    #[test]
    fn parse_errors_include_legacy_categories_and_byte_offsets() {
        let mut parser = Parser::new();
        assert_eq!(
            parser.parse("x+", "x", false).unwrap_err(),
            ParseError {
                kind: ParseErrorKind::Syntax,
                position: 2
            }
        );
        assert_eq!(
            parser.parse("missing+1", "", false).unwrap_err().kind,
            ParseErrorKind::UnknownIdentifier
        );
        assert_eq!(
            parser.parse("sin()", "", false).unwrap_err().kind,
            ParseErrorKind::InvalidFunctionParameters
        );
        assert_eq!(
            parser.parse("()", "", false).unwrap_err().kind,
            ParseErrorKind::EmptyParentheses
        );
        assert_eq!(
            parser.parse("x==y", "x,y", false).unwrap_err().kind,
            ParseErrorKind::Syntax
        );
    }

    #[test]
    fn variable_deduction_and_delimited_parse_work() {
        let mut parser = Parser::new();
        assert_eq!(
            parser
                .parse_and_deduce_variables("z+x+sin(y)", false)
                .unwrap(),
            ["x", "y", "z"]
        );
        close(parser.eval(&[2.0, 0.0, 3.0]).unwrap(), 5.0);
        assert_eq!(
            parser
                .parse_and_deduce_variables("a:=x+1;b:=a*2;b", false)
                .unwrap(),
            ["x"]
        );
        close(parser.eval(&[3.0]).unwrap(), 8.0);
        let error = parser
            .parse_and_deduce_variables("unknown(x)", false)
            .unwrap_err();
        assert_eq!(error.kind, ParseErrorKind::ExpectedOperator);
        assert_eq!(error.position, 7);

        parser.set_delimiter_char(Some(';'));
        parser.parse("2+3;ignored", "", false).unwrap();
        close(parser.eval(&[]).unwrap(), 5.0);
    }

    #[test]
    fn optimization_rewrites_trigonometric_square_sums() {
        let calls = Arc::new(AtomicU64::new(0));
        let callback_calls = Arc::clone(&calls);
        let mut parser = Parser::new();
        assert!(parser.add_function("count", 1, move |values| {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        parser
            .parse("sin(count(x))^2+cos(count(x))^2", "x", false)
            .unwrap();
        parser.optimize();
        close(parser.eval(&[0.5]).unwrap(), 1.0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn optimization_preserves_negative_zero_in_tanh_asinh_composition() {
        let mut parser = parse("tanh(asinh(x))", "x", false);
        parser.optimize();
        assert!(parser.eval(&[-0.0]).unwrap().is_sign_negative());
        close(parser.eval(&[2.0]).unwrap(), 2.0 / 5.0_f64.sqrt());
    }

    #[test]
    fn optimized_tan_acos_matches_legacy_division_error_at_zero() {
        let mut parser = parse("tan(acos(x))", "x", false);
        parser.optimize();
        assert_eq!(parser.eval(&[0.0]), Err(EvalError::DivisionByZero));
    }

    #[test]
    fn optimized_tan_abs_acos_matches_legacy_division_error_at_zero() {
        let mut parser = parse("tan(abs(acos(x)))", "x", false);
        parser.optimize();
        assert_eq!(parser.eval(&[0.0]), Err(EvalError::DivisionByZero));
        close(parser.eval(&[-0.5]).unwrap(), -3.0_f64.sqrt());
        close(parser.eval(&[0.5]).unwrap(), 3.0_f64.sqrt());
    }

    #[test]
    fn optimized_abs_tanh_matches_legacy_overflow_behavior() {
        let mut parser = parse("abs(tanh(x))", "x", false);
        parser.optimize();
        assert!(parser.eval(&[1.0e100]).unwrap().is_nan());

        for (expression, value) in [
            ("abs(tanh(cosh(x)))", 1.0e200),
            ("abs(tanh(exp(x)))", 1.0e200),
            ("abs(tanh(exp2(x)))", 1.0e200),
        ] {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            assert_eq!(parser.eval(&[value]).unwrap(), 1.0, "{expression}");
        }

        for (expression, expected) in [
            ("cos(abs(tanh(x)))", 1.0_f64.cos()),
            ("cosh(abs(tanh(x)))", 1.0_f64.cosh()),
        ] {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            assert_eq!(parser.eval(&[1.0e200]).unwrap(), expected, "{expression}");
            assert_eq!(parser.eval(&[-1.0e200]).unwrap(), expected, "{expression}");
        }
    }

    #[test]
    fn optimized_cosh_asinh_sinh_matches_legacy_overflow() {
        let mut parser = parse("cosh(asinh(sinh(x)))", "x", false);
        parser.optimize();
        close(parser.eval(&[-10.0]).unwrap(), 10.0_f64.cosh());
        assert_eq!(parser.eval(&[-1.0e200]).unwrap(), f64::INFINITY);
    }

    #[test]
    fn optimized_tanh_abs_matches_legacy_overflow_behavior() {
        let mut parser = parse("tanh(abs(x))", "x", false);
        parser.optimize();
        assert!(parser.eval(&[-1.0e200]).unwrap().is_nan());

        for expression in [
            "tanh(abs(cosh(x)))",
            "tanh(abs(exp(x)))",
            "tanh(abs(exp2(x)))",
        ] {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            for input in [10.0, 1.0e200] {
                assert_eq!(parser.eval(&[input]).unwrap(), 1.0, "{expression}");
            }
        }
    }

    #[test]
    fn optimized_tanh_acosh_cot_preserves_domain_error_order() {
        let mut parser = parse("tanh(acosh(cot(x)))", "x", false);
        parser.optimize();
        for input in [-2.0, -0.5, 2.0] {
            assert_eq!(
                parser.eval(&[input]),
                Err(EvalError::InvalidInverseTrigonometricInput),
                "x={input}"
            );
        }
        assert_eq!(parser.eval(&[-0.0]), Err(EvalError::DivisionByZero));
        assert_eq!(parser.eval(&[0.0]), Err(EvalError::DivisionByZero));
        let cotangent = 1.0 / 0.5_f64.tan();
        close(parser.eval(&[0.5]).unwrap(), cotangent.acosh().tanh());
    }

    #[test]
    fn optimized_cbrt_of_reciprocal_trig_matches_legacy_errors() {
        for (expression, input) in [
            ("cbrt(cot(acosh(x)))", 1.0e200),
            ("cbrt(csc(asinh(x)))", -1.0e200),
            ("cbrt(sec(exp(x)))", 1.0e200),
        ] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(
                parser.eval(&[input]),
                Err(EvalError::DivisionByZero),
                "{expression}"
            );
        }
    }

    #[test]
    fn optimized_trig_inverse_tanh_preserves_overflow_nan() {
        for expression in ["cot(asin(tanh(x)))", "tan(asin(tanh(x)))"] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            for input in [-1.0e200, 1.0e200] {
                assert!(
                    parser.eval(&[input]).unwrap().is_nan(),
                    "{expression} at {input}"
                );
            }
        }
    }

    #[test]
    fn optimized_inverse_trig_exponentials_preserve_underflow_behavior() {
        for (expression, expected) in [
            ("tan(acos(exp(x)))", f64::INFINITY),
            ("tan(acos(exp2(x)))", f64::INFINITY),
            ("cot(acos(exp(x)))", 0.0),
            ("cot(acos(exp2(x)))", 0.0),
            ("csc(asin(exp(x)))", f64::INFINITY),
            ("csc(asin(exp2(x)))", f64::INFINITY),
            ("sec(acos(exp(x)))", f64::INFINITY),
            ("sec(acos(exp2(x)))", f64::INFINITY),
        ] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(parser.eval(&[-1.0e200]).unwrap(), expected, "{expression}");
        }
    }

    #[test]
    fn optimization_preserves_zero_log_errors_through_inverse_trig() {
        for expression in [
            "exp(log10(acos(x)))",
            "exp(log2(acos(x)))",
            "exp2(log(acos(x)))",
            "exp2(log10(acos(x)))",
        ] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(
                parser.eval(&[1.0]),
                Err(EvalError::InvalidLogarithmOrPower),
                "{expression}"
            );
        }
    }

    #[test]
    fn optimization_matches_nested_log10_rounding() {
        let mut parser = parse("cbrt(log(log10(x)))", "x", false);
        parser.optimize();
        let expected = (10.0_f64.ln().ln() + std::f64::consts::LOG10_E.ln()).cbrt();
        close(parser.eval(&[10.0]).unwrap(), expected);

        let mut parser = parse("ceil(log(log10(x)))", "x", false);
        parser.optimize();
        assert_eq!(parser.eval(&[10.0]).unwrap(), 1.0);

        let mut parser = parse("cot(log(log10(x)))", "x", false);
        parser.optimize();
        assert!(parser.eval(&[10.0]).unwrap().is_finite());
    }

    #[test]
    fn optimization_matches_measured_log10_boundary_rewrites() {
        for (expression, value, expected) in [
            ("ceil(exp2(log10(x)))", 10.0, 3.0),
            ("floor(exp2(log10(x)))", -10.0, -3.0),
        ] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(parser.eval(&[value]).unwrap(), expected, "{expression}");
        }

        for expression in ["sinh(acosh(log10(x)))", "tanh(acosh(log10(x)))"] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            close(parser.eval(&[10.0]).unwrap(), 1.4901161193847656e-8);
        }
    }

    #[test]
    fn optimized_reciprocal_and_tangent_of_asin_cosh_match_legacy_rewrite() {
        for use_degrees in [false, true] {
            let angle = if use_degrees {
                90.0_f64.to_radians()
            } else {
                std::f64::consts::FRAC_PI_2
            };
            let expected = 0.5 / angle.cos();
            for expression in ["sec(asin(cosh(x)))", "tan(asin(cosh(x)))"] {
                let mut parser = parse(expression, "x", use_degrees);
                parser.optimize();
                for input in [-2.0_f64, -0.5, -0.0, 0.0, 0.5, 2.0] {
                    close(parser.eval(&[input]).unwrap(), expected);
                }
            }
        }
    }

    #[test]
    fn optimization_rewrites_atan_tan_to_atan2_branch_behavior() {
        let mut parser = parse("atan(tan(x))", "x", false);
        parser.optimize();
        for value in [-2.0, -1.0e200, 10.0] {
            close(
                parser.eval(&[value]).unwrap(),
                value.sin().atan2(value.cos()),
            );
        }
    }

    #[test]
    fn optimization_rewrites_comparisons_of_identical_expressions() {
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        let mut parser = Parser::new();
        assert!(parser.add_function("count", 1, move |values| {
            callback_calls.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        let cases = [
            ("count(x)=count(x)", 1.0),
            ("count(x)!=count(x)", 0.0),
            ("count(x)<count(x)", 0.0),
            ("count(x)<=count(x)", 1.0),
            ("count(x)>count(x)", 0.0),
            ("count(x)>=count(x)", 1.0),
        ];
        for (expression, expected) in cases {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            assert_eq!(parser.eval(&[2.0]).unwrap(), expected);
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn optimization_folds_domain_errors_like_the_legacy_optimizer() {
        let mut parser = parse("2+3*x+sin(0)", "x", false);
        parser.optimize();
        close(parser.eval(&[4.0]).unwrap(), 14.0);
        let mut parser = parse("log(0)+x", "x", false);
        parser.optimize();
        assert_eq!(parser.eval(&[0.0]).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn optimization_matches_observed_zero_absorber_rewrites() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_from_function = Arc::clone(&calls);
        let mut parser = Parser::new();
        assert!(parser.add_function("count", 1, move |values| {
            calls_from_function.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        parser.parse("0*count(x)", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 0.0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        parser.parse("0&count(x)", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 0.0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        parser.parse("0*log(-1)", "", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[]).unwrap(), 0.0);

        parser.parse("count(x)^0", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 1.0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        parser.parse("1^count(x)", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 1.0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        parser.parse("log(-1)^0", "", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[]).unwrap(), 1.0);

        parser.parse("1^log(-1)", "", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[]).unwrap(), 1.0);
    }

    #[test]
    fn optimization_matches_nested_inverse_reciprocal_rewrites() {
        let cases = [
            ("cos(abs(acos(x)))", -2.0, -2.0),
            ("cos(abs(acos(x)))", -0.5, -0.5),
            ("ceil(cot(atan(x)))", 0.5, 2.0),
            ("floor(cot(atan(x)))", -0.5, -2.0),
            ("cot(atan(cot(x)))", -0.5, (-0.5_f64).tan()),
            ("cot(atan(csc(x)))", -0.5, (-0.5_f64).sin()),
            ("csc(asin(cot(x)))", -0.5, (-0.5_f64).tan()),
            ("csc(asin(csc(x)))", -0.5, (-0.5_f64).sin()),
            ("sec(acos(cot(x)))", -0.5, (-0.5_f64).tan()),
            ("sec(acos(csc(x)))", -0.5, (-0.5_f64).sin()),
        ];
        for (expression, input, expected) in cases {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(parser.eval(&[input]).unwrap(), expected, "{expression}");
        }
        for expression in [
            "cot(atan(sqrt(x)))",
            "csc(asin(sqrt(x)))",
            "sec(acos(sqrt(x)))",
            "sqrt(csc(asin(x)))",
            "sqrt(sec(acos(x)))",
        ] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert!(parser.eval(&[-0.5]).unwrap().is_nan(), "{expression}");
            assert_eq!(parser.eval(&[0.0]), Err(EvalError::DivisionByZero));
        }
        let mut parser = parse("sqrt(cot(acos(x)))", "x", false);
        parser.optimize();
        assert!(parser.eval(&[-0.5]).unwrap().is_nan());
        assert_eq!(parser.eval(&[-0.0]), Err(EvalError::DivisionByZero));
        assert_eq!(parser.eval(&[0.0]), Err(EvalError::DivisionByZero));
    }

    #[test]
    fn optimization_matches_signed_nested_twelfth_root() {
        for input in [-2.0_f64, -0.5, -0.0, 0.0, 0.5, 2.0] {
            let expected = if input < 0.0 {
                -(-input).powf(1.0 / 12.0)
            } else {
                input.powf(1.0 / 12.0)
            };
            for expression in [
                "cbrt(sqrt(sqrt(x)))",
                "sqrt(sqrt(cbrt(x)))",
                "sqrt(cbrt(sqrt(x)))",
            ] {
                let mut parser = parse(expression, "x", false);
                parser.optimize();
                close(parser.eval(&[input]).unwrap(), expected);
            }
        }
    }

    #[test]
    fn optimization_rewrites_signed_sqrt_exp_log_compositions() {
        let cases = [
            ("sqrt(exp(log2(x)))", std::f64::consts::LOG2_E / 2.0),
            ("sqrt(exp2(log(x)))", std::f64::consts::LN_2 / 2.0),
            ("sqrt(exp2(log10(x)))", std::f64::consts::LOG10_2 / 2.0),
            ("sqrt(exp(log10(x)))", std::f64::consts::LOG10_E / 2.0),
        ];
        for (expression, exponent) in cases {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            for input in [-2.0_f64, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 10.0] {
                let magnitude = input.abs().powf(exponent);
                let expected = if input < 0.0 { -magnitude } else { magnitude };
                close(parser.eval(&[input]).unwrap(), expected);
            }
        }
    }

    #[test]
    fn optimization_rewrites_logarithms_of_exponential_roots() {
        let logarithms = [
            ("log", 1.0, std::f64::consts::LN_2),
            ("log2", std::f64::consts::LOG2_E, 1.0),
            (
                "log10",
                std::f64::consts::LOG10_E,
                std::f64::consts::LOG10_2,
            ),
        ];
        let roots = [("abs", 1.0), ("cbrt", 1.0 / 3.0), ("sqrt", 0.5)];
        for (logarithm, natural_factor, base_two_factor) in logarithms {
            for (root, root_factor) in roots {
                for (exponential, factor) in [("exp", natural_factor), ("exp2", base_two_factor)] {
                    let expression = format!("{logarithm}({root}({exponential}(x)))");
                    let mut parser = parse(&expression, "x", false);
                    parser.optimize();
                    let expected_factor = factor * root_factor;
                    for input in [-1.0e200_f64, -2.0, -0.0, 0.0, 0.5, 2.0, 1.0e200] {
                        close(parser.eval(&[input]).unwrap(), input * expected_factor);
                    }
                }
            }
        }
    }

    #[test]
    fn optimization_matches_measured_nested_log2_rounding() {
        let mut parser = parse("cbrt(log2(log2(x)))", "x", false);
        parser.optimize();
        close(parser.eval(&[2.0]).unwrap(), -4.310_406_785_344_659e-6);

        let mut parser = parse("sqrt(log2(log2(x)))", "x", false);
        parser.optimize();
        assert_eq!(parser.eval(&[2.0]), Err(EvalError::NegativeSquareRoot));

        for expression in ["int(log(log2(x)))", "int(log10(log2(x)))"] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            let result = parser.eval(&[2.0]).unwrap();
            assert_eq!(result, 0.0, "{expression}");
            assert!(!result.is_sign_negative(), "{expression}");
        }
    }

    #[test]
    fn optimization_matches_exp2_sqrt_integer_boundary() {
        for expression in ["floor(sqrt(exp2(x)))", "trunc(sqrt(exp2(x)))"] {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(parser.eval(&[2.0]).unwrap(), 1.0, "{expression}");
        }
    }

    #[test]
    fn optimization_preserves_measured_zero_log_errors() {
        let mut expressions = vec![
            "cbrt(cbrt(abs(x)))".to_owned(),
            "cbrt(sqrt(abs(x)))".to_owned(),
            "sqrt(cbrt(abs(x)))".to_owned(),
            "cbrt(cbrt(cot(x)))".to_owned(),
            "cbrt(cbrt(csc(x)))".to_owned(),
            "cbrt(sqrt(cot(x)))".to_owned(),
            "cbrt(sqrt(csc(x)))".to_owned(),
            "sqrt(cbrt(cot(x)))".to_owned(),
            "sqrt(cbrt(csc(x)))".to_owned(),
        ];
        for (outer, inner) in [
            ("exp2", "log"),
            ("exp2", "log10"),
            ("exp", "log2"),
            ("exp", "log10"),
        ] {
            for leaf in ["abs", "cot", "csc"] {
                expressions.push(format!("{outer}({inner}({leaf}(x)))"));
            }
        }
        for outer in ["log", "log2", "log10"] {
            for leaf in ["cot", "csc"] {
                expressions.push(format!("{outer}(abs({leaf}(x)))"));
            }
        }

        for expression in expressions {
            let mut parser = parse(&expression, "x", false);
            parser.optimize();
            for input in [-0.0, 0.0] {
                assert_eq!(
                    parser.eval(&[input]),
                    Err(EvalError::InvalidLogarithmOrPower),
                    "{expression} at {input}"
                );
            }
            assert!(parser.eval(&[1.0e-13]).is_ok(), "{expression} at 1e-13");
        }

        let mut zero_guard = parse("cbrt(cbrt(abs(x)))", "x", false);
        zero_guard.optimize();
        let mut negative_guard = parse("log(sqrt(cot(x)))", "x", false);
        negative_guard.optimize();
        let previous_epsilon = Parser::epsilon();
        Parser::set_epsilon(1.0);
        assert_eq!(
            zero_guard.eval(&[0.0]),
            Err(EvalError::InvalidLogarithmOrPower)
        );
        assert!(zero_guard.eval(&[1.0e-13]).is_ok());
        assert_eq!(
            negative_guard.eval(&[-0.5]),
            Err(EvalError::InvalidLogarithmOrPower)
        );
        Parser::set_epsilon(previous_epsilon);
    }

    #[test]
    fn optimization_preserves_legacy_negative_zero_in_log_trig_compositions() {
        let cases = [
            "log(sec(asin(x)))",
            "log(sec(atan(x)))",
            "log(sec(sin(x)))",
            "log(sec(tanh(x)))",
            "log(csc(acos(x)))",
            "log(abs(sec(x)))",
            "log10(sec(asin(x)))",
            "log10(sec(atan(x)))",
            "log10(sec(sin(x)))",
            "log10(sec(tanh(x)))",
            "log10(csc(acos(x)))",
            "log10(abs(sec(x)))",
            "log2(sec(asin(x)))",
            "log2(sec(atan(x)))",
            "log2(sec(sin(x)))",
            "log2(sec(tanh(x)))",
            "log2(csc(acos(x)))",
            "log2(abs(sec(x)))",
        ];
        for expression in cases {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            for input in [-0.0, 0.0, 1.0e-13] {
                assert_eq!(
                    parser.eval(&[input]).unwrap().to_bits(),
                    (-0.0_f64).to_bits(),
                    "{expression} at {input}"
                );
            }
        }
    }

    #[test]
    fn optimization_preserves_legacy_negative_zero_in_abs_sqrt_compositions() {
        let cases = [
            "abs(atan(sqrt(x)))",
            "abs(ceil(sqrt(x)))",
            "abs(floor(sqrt(x)))",
            "abs(sinh(sqrt(x)))",
            "abs(tanh(sqrt(x)))",
            "abs(trunc(sqrt(x)))",
        ];
        for expression in cases {
            let mut parser = parse(expression, "x", false);
            parser.optimize();
            assert_eq!(
                parser.eval(&[-0.0]).unwrap().to_bits(),
                (-0.0_f64).to_bits(),
                "{expression}"
            );
            assert_eq!(
                parser.eval(&[0.0]).unwrap().to_bits(),
                0.0_f64.to_bits(),
                "{expression}"
            );
            assert!(parser.eval(&[0.5]).unwrap() >= 0.0, "{expression}");
            assert_eq!(
                parser.eval(&[-0.5]),
                Err(EvalError::NegativeSquareRoot),
                "{expression}"
            );
        }
    }

    #[test]
    fn optimization_matches_nested_domain_optimizer_constants() {
        let cases = [
            ("acos(sec(cos(x)))", 0.0),
            ("asin(sec(cos(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(sec(asin(x)))", 0.0),
            ("asin(sec(asin(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(sec(atan(x)))", 0.0),
            ("asin(sec(atan(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(sec(sin(x)))", 0.0),
            ("asin(sec(sin(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(sec(tanh(x)))", 0.0),
            ("asin(sec(tanh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(exp(abs(x)))", 0.0),
            ("asin(exp(abs(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(exp(cbrt(x)))", 0.0),
            ("asin(exp(cbrt(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(exp(sqrt(x)))", 0.0),
            ("asin(exp(sqrt(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(exp2(abs(x)))", 0.0),
            ("asin(exp2(abs(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(exp2(cbrt(x)))", 0.0),
            ("asin(exp2(cbrt(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(exp2(sqrt(x)))", 0.0),
            ("asin(exp2(sqrt(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(abs(cosh(x)))", 0.0),
            ("asin(abs(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(cbrt(cosh(x)))", 0.0),
            ("asin(cbrt(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(ceil(cosh(x)))", 0.0),
            ("asin(ceil(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(floor(cosh(x)))", 0.0),
            ("asin(floor(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(int(cosh(x)))", 0.0),
            ("asin(int(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(trunc(cosh(x)))", 0.0),
            ("asin(trunc(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("acos(sqrt(cosh(x)))", 0.0),
            ("asin(sqrt(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("atan(atanh(cosh(x)))", std::f64::consts::FRAC_PI_2),
            ("sinh(acosh(cos(x)))", 0.0),
            ("sinh(acosh(sin(x)))", 0.0),
            ("sinh(acosh(tanh(x)))", 0.0),
            ("tanh(acosh(sin(x)))", 0.0),
            ("tanh(acosh(tanh(x)))", 0.0),
            ("tanh(atanh(cosh(x)))", 1.0),
        ];
        let mut parser = Parser::new();
        for (expression, expected) in cases {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            let actual = parser
                .eval(&[0.5])
                .unwrap_or_else(|error| panic!("{expression}: {error:?}"));
            assert_eq!(actual, expected, "{expression}");
        }
    }

    #[test]
    fn optimization_preserves_observed_signed_compositions() {
        let input: f64 = -0.5;
        let cot = 1.0 / input.tan();
        let csc = 1.0 / input.sin();
        let sec = 1.0 / input.cos();
        let cases = [
            ("abs(asinh(cbrt(x)))", input.cbrt().asinh()),
            ("abs(atan(cbrt(x)))", input.cbrt().atan()),
            ("abs(atanh(cbrt(x)))", input.cbrt().atanh()),
            ("abs(cbrt(asin(x)))", input.asin().cbrt().abs()),
            ("abs(cbrt(atan(x)))", input.atan().cbrt().abs()),
            ("abs(cbrt(x))", input.cbrt()),
            ("abs(cbrt(cbrt(x)))", input.cbrt().cbrt()),
            ("abs(cbrt(sqrt(x)))", -(-input).powf(1.0 / 6.0)),
            ("cbrt(sqrt(cot(x)))", -(-cot).powf(1.0 / 6.0)),
            ("cbrt(sqrt(csc(x)))", -(-csc).powf(1.0 / 6.0)),
            ("abs(cbrt(cos(x)))", input.cos().cbrt().abs()),
            ("abs(cbrt(cot(x)))", cot.cbrt().abs()),
            ("abs(cbrt(csc(x)))", csc.cbrt().abs()),
            ("abs(cbrt(sec(x)))", sec.cbrt().abs()),
            ("abs(cbrt(sin(x)))", input.sin().cbrt().abs()),
            ("abs(exp(log(x)))", input),
            ("abs(exp2(log2(x)))", input),
            ("abs(ceil(cbrt(x)))", input.cbrt().ceil()),
            ("abs(floor(cbrt(x)))", input.cbrt().floor()),
            ("abs(int(cbrt(x)))", input.cbrt().floor()),
            ("abs(sinh(cbrt(x)))", input.cbrt().sinh()),
            ("abs(tanh(cbrt(x)))", input.cbrt().tanh()),
            ("abs(trunc(cbrt(x)))", input.cbrt().trunc()),
            ("atan(tan(acos(x)))", input.acos()),
            ("sec(abs(acos(x)))", 1.0 / input),
        ];
        let mut parser = Parser::new();
        for (expression, expected) in cases {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            close(parser.eval(&[input]).unwrap(), expected);
        }
    }

    #[test]
    fn optimization_preserves_constant_signed_zero_before_rewriting_variables() {
        let mut parser = parse("0*-3", "", false);
        parser.optimize();
        assert!(parser.eval(&[]).unwrap().is_sign_negative());

        let mut parser = parse("0*x", "x", false);
        parser.optimize();
        assert_eq!(parser.eval(&[-3.0]).unwrap(), 0.0);
        assert!(!parser.eval(&[-3.0]).unwrap().is_sign_negative());

        parser.parse("-0*x", "x", false).unwrap();
        parser.optimize();
        assert!(parser.eval(&[3.0]).unwrap().is_sign_negative());

        parser.parse("x+0", "x", false).unwrap();
        parser.optimize();
        assert!(parser.eval(&[-0.0]).unwrap().is_sign_negative());

        parser.parse("0-x", "x", false).unwrap();
        parser.optimize();
        assert!(parser.eval(&[0.0]).unwrap().is_sign_negative());
    }

    #[test]
    fn absolute_value_of_square_root_matches_legacy_signed_zero_rewrite() {
        let mut parser = parse("abs(sqrt(x))", "x", false);
        assert!(parser.eval(&[-0.0]).unwrap().is_sign_negative());

        parser.optimize();
        assert!(parser.eval(&[-0.0]).unwrap().is_sign_negative());
    }

    #[test]
    fn optimization_applies_verified_identities_without_dropping_callbacks() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_from_function = Arc::clone(&calls);
        let mut parser = Parser::new();
        assert!(parser.add_function("count", 1, move |values| {
            calls_from_function.fetch_add(1, Ordering::SeqCst);
            values[0]
        }));
        parser.parse("sin(asin(count(x)))", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[2.0]).unwrap(), 2.0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        calls.store(0, Ordering::SeqCst);
        parser.parse("count(x)+count(x)", "x", false).unwrap();
        parser.optimize();
        parser.optimize();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 6.0);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        parser.parse("cos(acos(x))", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[2.0]).unwrap(), 2.0);

        parser.parse("exp(log(x))", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[-1.0]).unwrap(), -1.0);

        parser.parse("exp2(log2(x))", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[-1.0]).unwrap(), -1.0);

        parser.parse("tan(atan(x))", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[1.0e300]).unwrap(), 1.0e300);

        for (expression, expected) in [
            ("ceil(tan(atan(x)))", -2.0),
            ("floor(tan(atan(x)))", -2.0),
            ("trunc(tan(atan(x)))", -2.0),
        ] {
            parser.parse(expression, "x", false).unwrap();
            parser.optimize();
            assert_eq!(parser.eval(&[-2.0]).unwrap(), expected, "{expression}");
        }

        parser.parse("abs(tan(atan(x)))", "x", false).unwrap();
        parser.optimize();
        let large_input = 1.0e200_f64;
        assert_eq!(
            parser.eval(&[large_input]).unwrap(),
            large_input.atan().tan().abs()
        );

        parser.parse("sinh(asinh(x))", "x", false).unwrap();
        parser.optimize();
        assert_eq!(parser.eval(&[1.0e300]).unwrap(), 1.0e300);
    }

    #[test]
    fn identical_variable_division_matches_compile_time_rewrite() {
        let parser = parse("x/x", "x", false);
        assert_eq!(parser.eval(&[0.0]).unwrap(), 1.0);
        assert_eq!(parser.eval(&[2.0]).unwrap(), 1.0);

        let parser = parse("(x+1)/(x+1)", "x", false);
        assert_eq!(parser.eval(&[-1.0]).unwrap_err(), EvalError::DivisionByZero);
    }

    #[test]
    fn linked_parsers_evaluate_positionally_and_propagate_errors() {
        let mut square = Parser::new();
        square.parse("x^2", "x", false).unwrap();

        let mut sum = Parser::new();
        sum.parse("a+b", "a,b", false).unwrap();

        let mut negative_root = Parser::new();
        negative_root.parse("sqrt(-x)", "x", false).unwrap();

        let mut parser = Parser::new();
        assert!(parser.add_parser_function("square", Arc::new(square)));
        assert!(parser.add_parser_function("sum2", Arc::new(sum)));
        assert!(parser.add_parser_function("root", Arc::new(negative_root)));
        parser.parse("square(x)+sum2(x,2)", "x", false).unwrap();
        assert_eq!(parser.eval(&[3.0]).unwrap(), 14.0);
        parser.parse("root(x)", "x", false).unwrap();
        assert_eq!(
            parser.eval(&[1.0]).unwrap_err(),
            EvalError::NegativeSquareRoot
        );

        let unparsed = Parser::new();
        assert!(!parser.add_parser_function("unparsed", Arc::new(unparsed)));
    }
}
