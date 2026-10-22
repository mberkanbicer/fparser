#!/usr/bin/env python3
"""Compare a corpus against the pinned C++ parser and the Rust implementation."""

import math
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "differential_cases.tsv"
TOLERANCE = 1.0e-12
NONFINITE_VALUES = (-math.inf, math.inf, math.nan)
FLOAT_BOUNDARY_VALUES = (
    -1.7976931348623157e308,
    -2.2250738585072014e-308,
    -4.9406564584124654e-324,
    4.9406564584124654e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
)
UNARY_FUNCTIONS = (
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cbrt",
    "ceil",
    "cos",
    "cosh",
    "cot",
    "csc",
    "exp",
    "exp2",
    "floor",
    "int",
    "log",
    "log10",
    "log2",
    "sec",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
)
TRIPLE_AUDIT_VALUES = (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0)
TRIPLE_STRESS_VALUES = (
    -1.0e200,
    -10.0,
    -2.0,
    -1.0,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.0,
    2.0,
    10.0,
    1.0e200,
)


def generate_operator_cases():
    values = (
        -2.0,
        -0.5,
        -0.49,
        -0.0625,
        -0.0,
        0.0,
        0.0625,
        0.49,
        0.5,
        1.0,
        0.06250000000001,
        0.0625000000001,
        2.0,
    ) + FLOAT_BOUNDARY_VALUES + NONFINITE_VALUES
    operators = (
        ("add", "+"),
        ("subtract", "-"),
        ("multiply", "*"),
        ("divide", "/"),
        ("modulo", "%"),
        ("power", "^"),
        ("equal", "="),
        ("not_equal", "!="),
        ("less", "<"),
        ("less_equal", "<="),
        ("greater", ">"),
        ("greater_equal", ">="),
        ("and", "&"),
        ("or", "|"),
    )

    def format_value(value):
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0"
        return format(value, ".17g")

    rows = []
    for optimize in (False, True):
        variables = "@optimize:x,y" if optimize else "x,y"
        suffix = "optimized" if optimize else "plain"
        for operator_name, operator in operators:
            for left_index, left in enumerate(values):
                for right_index, right in enumerate(values):
                    case_id = (
                        f"generated_{suffix}_{operator_name}_"
                        f"{left_index}_{right_index}"
                    )
                    rows.append(
                        "\t".join(
                            (
                                case_id,
                                f"(x{operator}y)",
                                variables,
                                "0",
                                f"{format_value(left)},{format_value(right)}",
                            )
                        )
                    )
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                rows.append(
                    "\t".join(
                        (
                            f"generated_{suffix}_builtin_power_{left_index}_{right_index}",
                            "pow(x,y)",
                            variables,
                            "0",
                            f"{format_value(left)},{format_value(right)}",
                        )
                    )
                )
    for optimize in (False, True):
        variables = "@optimize:x" if optimize else "x"
        suffix = "optimized" if optimize else "plain"
        for value_index, value in enumerate(values):
            rows.append(
                "\t".join(
                    (
                        f"generated_{suffix}_not_{value_index}",
                        "!x",
                        variables,
                        "0",
                        format_value(value),
                    )
                )
            )
        for operator_name, operator in operators:
            for value_index, value in enumerate(values):
                rows.append(
                    "\t".join(
                        (
                            f"generated_{suffix}_callback_{operator_name}_{value_index}",
                            f"(count(x){operator}count(x))",
                            variables,
                            "0",
                            format_value(value),
                        )
                    )
                )
    return "\n".join(rows)


def generate_builtin_cases():
    unary_values = {
        "abs": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "acos": (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
        "acosh": (0.0, 1.0, 1.5, 2.0),
        "asin": (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
        "asinh": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "atan": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "atanh": (-1.0, -0.5, 0.0, 0.5, 1.0),
        "cbrt": (-2.0, -0.49, -0.0, 0.0, 0.49, 2.0),
        "ceil": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "cos": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "cosh": (-2.0, -0.5, 0.0, 0.5, 2.0),
        "cot": (-2.0, -0.5, 0.5, 2.0),
        "csc": (-2.0, -0.5, 0.5, 2.0),
        "exp": (-10.0, -0.5, -0.0, 0.0, 0.5, 10.0),
        "exp2": (-10.0, -0.49, -0.0, 0.0, 0.49, 4.0, 10.0),
        "floor": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "int": (-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0),
        "log": (-1.0, 0.0, 0.01, 0.5, 1.0, 2.0, 10.0),
        "log10": (-1.0, 0.0, 0.01, 0.5, 1.0, 2.0, 10.0),
        "log2": (-1.0, 0.0, 0.01, 0.5, 1.0, 2.0, 10.0),
        "sec": (-2.0, -0.5, 0.5, 2.0),
        "sin": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "sinh": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "sqrt": (-1.0, 0.0, 0.5, 2.0),
        "tan": (-2.0, -0.5, 0.5, 2.0),
        "tanh": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
        "trunc": (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0),
    }
    boundary_unary_values = {
        "abs": FLOAT_BOUNDARY_VALUES,
        "acosh": (1.0000000000000002, 1.7976931348623157e308),
        "atanh": (-0.9999999999999999, 0.9999999999999999),
        "cbrt": FLOAT_BOUNDARY_VALUES,
        "ceil": FLOAT_BOUNDARY_VALUES,
        "cosh": (-20.0, 20.0, 710.0),
        "exp": (-745.1332191019412, -745.1332191019411, 709.782712893384, 709.7827128933841),
        "exp2": (-1075.0, -1074.0, 1023.0, 1024.0),
        "floor": FLOAT_BOUNDARY_VALUES,
        "int": FLOAT_BOUNDARY_VALUES,
        "log": (4.9406564584124654e-324, 2.2250738585072014e-308, 1.7976931348623157e308),
        "log10": (4.9406564584124654e-324, 2.2250738585072014e-308, 1.7976931348623157e308),
        "log2": (4.9406564584124654e-324, 2.2250738585072014e-308, 1.7976931348623157e308),
        "sinh": (-710.0, -20.0, 20.0, 710.0),
        "sqrt": (-1.7976931348623157e308, -4.9406564584124654e-324)
        + FLOAT_BOUNDARY_VALUES[2:],
        "trunc": FLOAT_BOUNDARY_VALUES,
    }
    binary_values = (
        -1.0e200,
        -2.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        2.0,
        1.0e200,
    ) + FLOAT_BOUNDARY_VALUES + NONFINITE_VALUES

    def format_value(value):
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0"
        return format(value, ".17g")

    rows = []
    for optimize in (False, True):
        suffix = "optimized" if optimize else "plain"
        variable_prefix = "@optimize:" if optimize else ""
        for name, values in unary_values.items():
            for index, value in enumerate(
                values + boundary_unary_values.get(name, ()) + NONFINITE_VALUES
            ):
                rows.append(
                    "\t".join(
                        (
                            f"generated_{suffix}_{name}_{index}",
                            f"{name}(x)",
                            f"{variable_prefix}x",
                            "0",
                            format_value(value),
                        )
                    )
                )
        for name in ("atan2", "hypot", "max", "min"):
            for left_index, left in enumerate(binary_values):
                for right_index, right in enumerate(binary_values):
                    rows.append(
                        "\t".join(
                            (
                                f"generated_{suffix}_{name}_{left_index}_{right_index}",
                                f"{name}(x,y)",
                                f"{variable_prefix}x,y",
                                "0",
                                f"{format_value(left)},{format_value(right)}",
                            )
                        )
                    )
    return "\n".join(rows)


def generate_nested_builtin_cases():
    values = (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0)

    def format_value(value):
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0"
        return format(value, ".17g")

    rows = []
    for outer in UNARY_FUNCTIONS:
        for inner in UNARY_FUNCTIONS:
            for index, value in enumerate(values):
                rows.append(
                    "\t".join(
                        (
                            f"generated_plain_{outer}_{inner}_{index}",
                            f"{outer}({inner}(x))",
                            "x",
                            "0",
                            format_value(value),
                        )
                    )
                )
    for outer in UNARY_FUNCTIONS:
        for inner in UNARY_FUNCTIONS:
            for index, value in enumerate(values):
                rows.append(
                    "\t".join(
                        (
                            f"generated_optimized_{outer}_{inner}_{index}",
                            f"{outer}({inner}(x))",
                            "@optimize:x",
                            "0",
                            format_value(value),
                        )
                    )
                )
    rows.append(
        "\t".join(
            (
                "generated_optimized_abs_tanh_extreme",
                "abs(tanh(x))",
                "@optimize:x",
                "0",
                "1e100",
            )
        )
    )
    stress_values = (
        -1.0e200,
        -2.0,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        2.0,
        10.0,
        1.0e100,
        1.0e200,
    )
    for outer in UNARY_FUNCTIONS:
        for inner in UNARY_FUNCTIONS:
            for index, value in enumerate(stress_values):
                rows.append(
                    "\t".join(
                        (
                            f"generated_stress_optimized_{outer}_{inner}_{index}",
                            f"{outer}({inner}(x))",
                            "@optimize:x",
                            "0",
                            format_value(value),
                        )
                    )
                )
    for outer in UNARY_FUNCTIONS:
        for inner in UNARY_FUNCTIONS:
            for index, value in enumerate(stress_values):
                rows.append(
                    "\t".join(
                        (
                            f"generated_stress_plain_{outer}_{inner}_{index}",
                            f"{outer}({inner}(x))",
                            "x",
                            "0",
                            format_value(value),
                        )
                    )
                )
    return "\n".join(rows)


def generate_triple_optimizer_cases(values):
    def format_value(value):
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0"
        return format(value, ".17g")

    rows = []
    for outer in UNARY_FUNCTIONS:
        for middle in UNARY_FUNCTIONS:
            for inner in UNARY_FUNCTIONS:
                expression = f"{outer}({middle}({inner}(x)))"
                for index, value in enumerate(values):
                    rows.append(
                        "\t".join(
                            (
                                f"generated_triple_{outer}_{middle}_{inner}_{index}",
                                expression,
                                "@optimize:x",
                                "0",
                                format_value(value),
                            )
                        )
                    )
    return "\n".join(rows)


def generate_nested_optimizer_regressions():
    logarithms = ("log", "log2", "log10")
    roots = ("abs", "cbrt", "sqrt")
    exponentials = ("exp", "exp2")
    values = (-1.0e200, -2.0, -0.0, 0.0, 0.5, 2.0, 1.0e200)

    def format_value(value):
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return "-0"
        return format(value, ".17g")

    rows = []
    for logarithm in logarithms:
        for root in roots:
            for exponential in exponentials:
                for index, value in enumerate(values):
                    expression = f"{logarithm}({root}({exponential}(x)))"
                    case_id = (
                        f"generated_optimizer_log_{logarithm}_{root}_"
                        f"{exponential}_{index}"
                    )
                    rows.append(
                        "\t".join(
                            (
                                case_id,
                                expression,
                                "@optimize:x",
                                "0",
                                format_value(value),
                            )
                        )
                    )
    return "\n".join(rows)


def generate_optimizer_zero_error_cases():
    patterns = [
        "cbrt(cbrt(abs(x)))",
        "cbrt(sqrt(abs(x)))",
        "sqrt(cbrt(abs(x)))",
    ]
    for outer, inner in (
        ("cbrt", "cbrt"),
        ("cbrt", "sqrt"),
        ("sqrt", "cbrt"),
    ):
        for leaf in ("cot", "csc"):
            patterns.append(f"{outer}({inner}({leaf}(x)))")
    for outer, inner in (
        ("exp2", "log"),
        ("exp2", "log10"),
        ("exp", "log2"),
        ("exp", "log10"),
    ):
        for leaf in ("abs", "cot", "csc"):
            patterns.append(f"{outer}({inner}({leaf}(x)))")
    for outer in ("log", "log2", "log10"):
        for leaf in ("cot", "csc"):
            patterns.append(f"{outer}(abs({leaf}(x)))")

    values = (-0.0, 0.0, 1.0e-13)
    rows = []
    for pattern_index, expression in enumerate(patterns):
        for value_index, value in enumerate(values):
            value_text = (
                "-0"
                if value == 0.0 and math.copysign(1.0, value) < 0
                else format(value, ".17g")
            )
            rows.append(
                "\t".join(
                    (
                        f"generated_optimizer_zero_error_{pattern_index}_{value_index}",
                        expression,
                        "@optimize:x",
                        "0",
                        value_text,
                    )
                )
            )
    return "\n".join(rows)


def generate_optimizer_signed_exp_log_cases():
    values = (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0)
    rows = []
    for pattern_index, (outer, inner) in enumerate(
        (
            ("exp2", "log"),
            ("exp2", "log10"),
            ("exp", "log2"),
            ("exp", "log10"),
        )
    ):
        for leaf in ("cot", "csc"):
            for value_index, value in enumerate(values):
                value_text = (
                    "-0"
                    if value == 0.0 and math.copysign(1.0, value) < 0
                    else format(value, ".17g")
                )
                rows.append(
                    "\t".join(
                        (
                            f"generated_optimizer_signed_exp_log_{pattern_index}_{leaf}_{value_index}",
                            f"{outer}({inner}({leaf}(x)))",
                            "@optimize:x",
                            "0",
                            value_text,
                        )
                    )
                )
    return "\n".join(rows)


def generate_optimizer_negative_sqrt_log_cases():
    values = (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0)
    rows = []
    for outer in ("log", "log2", "log10"):
        for leaf in ("cot", "csc", "sec"):
            for value_index, value in enumerate(values):
                value_text = (
                    "-0"
                    if value == 0.0 and math.copysign(1.0, value) < 0
                    else format(value, ".17g")
                )
                rows.append(
                    "\t".join(
                        (
                            f"generated_optimizer_negative_sqrt_log_{outer}_{leaf}_{value_index}",
                            f"{outer}(sqrt({leaf}(x)))",
                            "@optimize:x",
                            "0",
                            value_text,
                        )
                    )
                )
    return "\n".join(rows)


def generate_optimizer_signed_nested_composition_cases():
    patterns = (
        "sqrt(cbrt(cot(x)))",
        "sqrt(cbrt(csc(x)))",
        "cosh(asinh(sqrt(x)))",
        "sec(acos(cot(x)))",
        "sec(acos(csc(x)))",
        "sec(acos(sqrt(x)))",
        "sqrt(csc(asin(x)))",
        "sqrt(sec(acos(x)))",
        "sqrt(cot(acos(x)))",
        "tanh(acosh(cot(x)))",
        "sec(asin(cosh(x)))",
        "tan(asin(cosh(x)))",
    )
    values = (-2.0, -0.5, -0.0, 0.0, 0.5, 2.0)
    rows = []
    for pattern_index, expression in enumerate(patterns):
        for value_index, value in enumerate(values):
            value_text = (
                "-0"
                if value == 0.0 and math.copysign(1.0, value) < 0
                else format(value, ".17g")
            )
            rows.append(
                "\t".join(
                    (
                        f"generated_optimizer_signed_nested_{pattern_index}_{value_index}",
                        expression,
                        "@optimize:x",
                        "0",
                        value_text,
                    )
                )
            )
    for pattern_index, expression in enumerate(patterns[-2:], start=len(patterns) - 2):
        for value_index, value in enumerate(values):
            value_text = (
            "-0"
            if value == 0.0 and math.copysign(1.0, value) < 0
            else format(value, ".17g")
            )
            rows.append(
            "\t".join(
                (
                    f"generated_optimizer_signed_nested_degrees_{pattern_index}_{value_index}",
                    expression,
                    "@optimize:x",
                    "1",
                    value_text,
                )
            )
            )
    return "\n".join(rows)


def run(command):
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout.splitlines()


def parse_records(lines):
    records = {}
    for line in lines:
        case_id, status, value, code = line.split("\t")
        if case_id in records:
            raise ValueError(f"duplicate result for {case_id}")
        records[case_id] = (status, value, int(code))
    return records


def parse_float(value):
    try:
        return float(value)
    except ValueError:
        if "nan" in value.lower():
            return math.nan
        raise


def compare(reference, rust, collect_mismatches=False):
    if reference.keys() != rust.keys():
        raise AssertionError(
            f"case IDs differ: C++={sorted(reference)}, Rust={sorted(rust)}"
        )
    mismatches = []

    def check_matches(matches, message):
        if matches:
            return
        if not collect_mismatches:
            raise AssertionError(message)
        mismatches.append(message)

    for case_id, (reference_status, reference_value, reference_code) in reference.items():
        rust_status, rust_value, rust_code = rust[case_id]
        if (reference_status, reference_code) != (rust_status, rust_code):
            check_matches(
                False,
                f"{case_id}: C++ status/code "
                f"{reference_status}/{reference_code}, Rust "
                f"{rust_status}/{rust_code}",
            )
            continue
        if reference_status == "parse":
            check_matches(
                reference_value == rust_value,
                f"{case_id}: parse offset C++={reference_value}, Rust={rust_value}",
            )
        elif reference_status == "vars":
            check_matches(
                reference_value == rust_value,
                f"{case_id}: variables C++={reference_value}, Rust={rust_value}",
            )
        elif reference_status == "calls":
            check_matches(
                reference_value == rust_value,
                f"{case_id}: callback count C++={reference_value}, Rust={rust_value}",
            )
        elif reference_code == 0:
            expected = parse_float(reference_value)
            actual = parse_float(rust_value)
            if math.isnan(expected):
                matches = math.isnan(actual)
            elif math.isinf(expected):
                matches = expected == actual
            elif expected == 0.0 and actual == 0.0:
                matches = math.copysign(1.0, expected) == math.copysign(1.0, actual)
            else:
                matches = abs(expected - actual) <= TOLERANCE * max(
                    1.0, abs(expected), abs(actual)
                )
            check_matches(
                matches, f"{case_id}: C++={reference_value}, Rust={rust_value}"
            )
    case_count = sum(status != "calls" for status, _, _ in reference.values())
    if mismatches:
        mismatch_ids = [message.split(":", 1)[0] for message in mismatches]
        pattern_count = len({case_id.rsplit("_", 1)[0] for case_id in mismatch_ids})
        print(
            f"Differential comparison found {len(mismatches)} mismatches "
            f"across {pattern_count} patterns in {case_count} expressions"
        )
        for mismatch in mismatches:
            print(mismatch)
        return mismatches
    print(
        f"Differential comparison passed: {case_count} expressions and "
        f"{len(reference) - case_count} callback-count checks"
    )
    return mismatches


def main():
    arguments = sys.argv[1:]
    triple_audit = arguments in (
        ["--triple-audit"],
        ["--triple-stress-audit"],
    )
    if arguments and not triple_audit:
        raise SystemExit(
            "usage: run_differential.py [--triple-audit | --triple-stress-audit]"
        )
    with tempfile.TemporaryDirectory(prefix="fparser-rust-reference-") as build_dir:
        build_dir = pathlib.Path(build_dir)
        corpus_path = build_dir / "differential_cases.tsv"
        if triple_audit:
            values = (
                TRIPLE_STRESS_VALUES
                if arguments == ["--triple-stress-audit"]
                else TRIPLE_AUDIT_VALUES
            )
            corpus = generate_triple_optimizer_cases(values)
        else:
            corpus = "\n".join(
                (
                    CORPUS.read_text(encoding="utf-8"),
                    generate_operator_cases(),
                    generate_builtin_cases(),
                    generate_nested_builtin_cases(),
                    generate_nested_optimizer_regressions(),
                    generate_optimizer_zero_error_cases(),
                    generate_optimizer_signed_exp_log_cases(),
                    generate_optimizer_negative_sqrt_log_cases(),
                    generate_optimizer_signed_nested_composition_cases(),
                )
            )
        corpus_path.write_text(corpus, encoding="utf-8")
        run(
            ["cmake", "-S", str(ROOT / "tests"), "-B", str(build_dir)],
        )
        run(
            ["cmake", "--build", str(build_dir), "--config", "Release"],
        )
        executable_name = (
            "fparser-reference.exe" if sys.platform == "win32" else "fparser-reference"
        )
        executables = list(build_dir.rglob(executable_name))
        if not executables:
            raise RuntimeError(f"CMake did not produce {executable_name}")
        executable = executables[0]
        reference = run([str(executable), str(corpus_path)])
        rust = run(
            [
                "cargo",
                "run",
                "--manifest-path",
                str(ROOT / "Cargo.toml"),
                "--release",
                "--quiet",
                "--bin",
                "differential_cases",
                "--",
                str(corpus_path),
            ]
        )
    mismatches = compare(
        parse_records(reference),
        parse_records(rust),
        collect_mismatches=triple_audit,
    )
    if triple_audit and mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
