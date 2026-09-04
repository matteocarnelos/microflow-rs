//! The week-3 safety net (spec §5.3): host inference of the real
//! `conv1d.tflite` via `#[model]` against the TFLite interpreter, within
//! ±1 quantum per output element (the requant operation order may differ
//! between implementations — `round_ties_even` here vs the reference kernel).
//!
//! The fixtures are produced by the Python side:
//!
//! ```text
//! tmp/venv312/bin/python ml/scripts/dump_parity_fixtures.py
//! ```
//!
//! (from the repository root; the script runs the tf.lite.Interpreter on
//! deterministic windows and writes `tests/golden/parity/conv1d.txt`).
//! Without the fixture file the test skips: it cannot fabricate reference
//! outputs without the interpreter.

use std::fs;

use microflow_macros::model;
use nalgebra::SMatrix;

#[model("models/conv1d.tflite")]
struct Conv1DSpike;

const FIXTURE: &str = "tests/golden/parity/conv1d.txt";

struct Case {
    input: [i8; 128],
    expected_output: [i8; 4],
}

/// Parses the fixture: a header line with `input_scale input_zp output_scale
/// output_zp`, then per case an `input` section of 128 lines, an `output`
/// section of 4 lines, one int8 per line.
fn parse_fixture(text: &str) -> Option<(f32, i8, f32, i8, Vec<Case>)> {
    let mut lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'));
    let header: Vec<f32> = lines
        .next()?
        .split_whitespace()
        .filter_map(|token| token.parse::<f32>().ok())
        .collect();
    if header.len() != 4 {
        return None;
    }
    let (input_scale, input_zero_point, output_scale, output_zero_point) =
        (header[0], header[1] as i8, header[2], header[3] as i8);

    let mut cases = Vec::new();
    let mut case = Case {
        input: [0; 128],
        expected_output: [0; 4],
    };
    let mut filled = 0usize;
    let mut in_input = false;
    let mut in_output = false;
    for line in lines {
        match line {
            "input" => {
                if in_output {
                    if filled != case.expected_output.len() {
                        return None;
                    }
                    cases.push(case);
                    case = Case {
                        input: [0; 128],
                        expected_output: [0; 4],
                    };
                }
                in_input = true;
                in_output = false;
                filled = 0;
            }
            "output" => {
                if !in_input || filled != case.input.len() {
                    return None;
                }
                in_input = false;
                in_output = true;
                filled = 0;
            }
            value => {
                let value: i8 = value.parse().ok()?;
                if in_input && filled < case.input.len() {
                    case.input[filled] = value;
                    filled += 1;
                } else if in_output && filled < case.expected_output.len() {
                    case.expected_output[filled] = value;
                    filled += 1;
                } else {
                    return None;
                }
            }
        }
    }
    if in_output && filled == case.expected_output.len() {
        cases.push(case);
    }
    Some((
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        cases,
    ))
}

/// Self-check of the harness (no interpreter involved): a fixture built from
/// the model's own output must parse and compare with a zero diff — the
/// output scale is a power of two, so the quantized recovery round-trips
/// exactly.
#[test]
fn fixture_parser_roundtrip() {
    let input: SMatrix<i8, 128, 1> = SMatrix::from_element(0);
    let output = Conv1DSpike::predict_quantized(input);
    let output_scale = 0.00390625f32;
    let output_zero_point = -128i8;
    let quantized: Vec<i8> = output
        .iter()
        .map(|v| (v / output_scale).round() as i32 + output_zero_point as i32)
        .map(|v| v.clamp(-128, 127) as i8)
        .collect();

    let mut fixture = String::from("0.012819952 1 0.00390625 -128\n");
    fixture.push_str("input\n");
    for _ in 0..128 {
        fixture.push_str("0\n");
    }
    fixture.push_str("output\n");
    for q in &quantized {
        fixture.push_str(&format!("{q}\n"));
    }

    let Some((_, _, scale, zp, cases)) = parse_fixture(&fixture) else {
        panic!("the parser rejected a well-formed fixture");
    };
    assert_eq!((scale, zp), (output_scale, output_zero_point));
    assert_eq!(cases.len(), 1);
    assert_eq!(cases[0].input, [0; 128]);
    assert_eq!(cases[0].expected_output, quantized.as_slice());
}

#[test]
fn conv1d_matches_tflite_interpreter_within_one_quantum() {
    let Ok(text) = fs::read_to_string(FIXTURE) else {
        eprintln!("SKIPPED: {FIXTURE} not found — run ml/scripts/dump_parity_fixtures.py first");
        return;
    };
    let Some((_, _, output_scale, output_zero_point, cases)) = parse_fixture(&text) else {
        panic!("could not parse {FIXTURE}");
    };
    assert!(
        !cases.is_empty(),
        "{FIXTURE} holds no cases — regenerate it"
    );

    for (n, case) in cases.iter().enumerate() {
        let input: SMatrix<i8, 128, 1> = SMatrix::from_fn(|t, _| case.input[t]);
        let output = Conv1DSpike::predict_quantized(input);
        // Recover the quantized output: dequantize is (q - zp) * scale, so
        // q = out / scale + zp (exact up to one rounding — covered by the
        // ±1 quantum tolerance).
        let quantized: Vec<i8> = output
            .iter()
            .map(|v| (v / output_scale).round() as i32 + output_zero_point as i32)
            .map(|v| v.clamp(-128, 127) as i8)
            .collect();
        let argmax = |values: &[i8]| {
            values
                .iter()
                .enumerate()
                .max_by_key(|(i, v)| (*v, usize::MAX - *i))
                .map(|(i, _)| i)
                .unwrap_or(usize::MAX)
        };
        assert_eq!(
            argmax(&quantized),
            argmax(&case.expected_output),
            "case {n}: argmax differs"
        );
        for (k, (got, expected)) in quantized.iter().zip(case.expected_output).enumerate() {
            let diff = (*got as i32 - expected as i32).abs();
            assert!(
                diff <= 1,
                "case {n}, output {k}: {got} vs interpreter {expected} (diff {diff} > 1 quantum)"
            );
        }
    }
}
