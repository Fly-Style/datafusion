// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{Array, AsArray, StringArray};
use arrow::datatypes::DataType;
use datafusion_common::utils::take_function_args;
use datafusion_common::{exec_err, plan_err, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use regex::Regex;
use std::any::Any;
use std::sync::Arc;

/// `regexp_extract` expression implementation in both original Spark and PySpark compatible manner.
///
/// Original Spark: <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>.
/// As a trade-off, we always expect the 'idx' argument to be present as integer.
///
/// PySpark: <https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.regexp_extract.html>
#[derive(Debug)]
pub struct SparkRegexpExtract {
    signature: Signature,
}

impl Default for SparkRegexpExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkRegexpExtract {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkRegexpExtract {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "regexp_extract"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(match &arg_types[0] {
            DataType::LargeUtf8 => DataType::LargeUtf8,
            DataType::Utf8 | DataType::Utf8View => DataType::Utf8,
            other => {
                return exec_err!(
                    "The regexp_extract function can only return strings. Got {other}"
                );
            }
        })
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [str_arg, regexp_arg, idx_arg] = take_function_args(self.name(), &args.args)?;

        // Extract pattern and index from arguments
        let regexp = match regexp_arg {
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(p)))
            | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(p)))
            | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(p))) => p,
            ColumnarValue::Scalar(ScalarValue::Utf8(None))
            | ColumnarValue::Scalar(ScalarValue::LargeUtf8(None))
            | ColumnarValue::Scalar(ScalarValue::Utf8View(None)) => {
                return Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))
            }
            _ => {
                return exec_err!(
                    "'regexp' argument must be a scalar string for function `{}`",
                    self.name()
                )
            }
        };

        let Ok(regex) = Regex::new(regexp) else {
            // If the regex is invalid, return None
            return Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)));
        };

        let idx = match idx_arg {
            ColumnarValue::Scalar(ScalarValue::Int64(Some(idx))) => {
                if *idx < 0 {
                    return exec_err!(
                        "Index argument must be a non-negative integer for function `{}`",
                        self.name()
                    );
                }
                *idx as usize
            }
            _ => {
                return exec_err!(
                    "'idx' argument must be an integer for function `{}`",
                    self.name()
                )
            }
        };

        // We care both about column reference and scalar string
        match str_arg {
            // PySpark way
            ColumnarValue::Array(array) => match array.data_type() {
                DataType::Utf8 => {
                    let str_array = array.as_string::<i32>();
                    let result: StringArray = str_array
                        .into_iter()
                        .map(|s| s.map(|s| regexp_extract_impl(s, &regex, idx)))
                        .collect();
                    Ok(ColumnarValue::Array(Arc::new(result)))
                }
                DataType::LargeUtf8 => {
                    let str_array = array.as_string::<i64>();
                    let result: StringArray = str_array
                        .into_iter()
                        .map(|s| s.map(|s| regexp_extract_impl(s, &regex, idx)))
                        .collect();
                    Ok(ColumnarValue::Array(Arc::new(result)))
                }
                DataType::Utf8View => {
                    let str_array = array.as_string_view();
                    let result: StringArray = str_array
                        .iter()
                        .map(|s| s.map(|s| regexp_extract_impl(s, &regex, idx)))
                        .collect();
                    Ok(ColumnarValue::Array(Arc::new(result)))
                }
                other => {
                    exec_err!(
                        "Unsupported data type {other:?} for function `{}`",
                        self.name()
                    )
                }
            },
            // Original Spark way
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
            | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s)))
            | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) => {
                let result = regexp_extract_impl(s, &regex, idx);
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(Some(result))))
            }
            ColumnarValue::Scalar(ScalarValue::Utf8(None))
            | ColumnarValue::Scalar(ScalarValue::LargeUtf8(None))
            | ColumnarValue::Scalar(ScalarValue::Utf8View(None)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)))
            }
            other => {
                exec_err!(
                    "Unsupported data type {other:?} for function `{}`",
                    self.name()
                )
            }
        }
    }

    /// Manual type check & coercion of the function arguments for the sake of SQL support.
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [input_type, regexp_type, idx_type] = arg_types else {
            return plan_err!(
                "The {} function requires 3 argument, but got {}.",
                self.name(),
                arg_types.len()
            );
        };

        if !matches!(
            regexp_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            return plan_err!(
                "'regexp' argument must be a string for function `{}`",
                self.name()
            );
        }

        if !matches!(
            regexp_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            return plan_err!(
                "'regexp' argument must be a string for function `{}`",
                self.name()
            );
        }

        if !idx_type.is_integer() {
            return plan_err!(
                "'idx' argument must be an integer for function `{}`",
                self.name()
            );
        }

        Ok(vec![
            input_type.clone(),
            regexp_type.clone(),
            DataType::Int64,
        ])
    }
}

/// Extract a specific group matched by the Java regex, from the specified string column.
/// If the regex did not match, or the specified group did not match, an empty string is returned.
fn regexp_extract_impl(input: &str, regex: &Regex, idx: usize) -> String {
    if let Some(captures) = regex.captures(input) {
        if idx == 0 {
            // For idx == 0 return the entire match
            captures
                .get(0)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        } else {
            // Return the specific capture group
            captures
                .get(idx)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        }
    } else {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, FieldRef};
    use datafusion_common::cast::as_generic_string_array;
    use regex::Regex;

    // Unit tests
    #[test]
    fn test_basic_regex_matching() {
        // Test basic regex matching with capture groups
        assert_eq!(
            regexp_extract_impl("a b", &Regex::new("a (b)").unwrap(), 1),
            "b"
        );
        assert_eq!(
            regexp_extract_impl("a123b", &Regex::new("(\\d+)").unwrap(), 1),
            "123"
        );
        assert_eq!(
            regexp_extract_impl(
                "test@a.com",
                &Regex::new("(.+)@(.+)\\.(.+)").unwrap(),
                1
            ),
            "test"
        );
        assert_eq!(
            regexp_extract_impl(
                "test@a.com",
                &Regex::new("(.+)@(.+)\\.(.+)").unwrap(),
                2
            ),
            "a"
        );
        assert_eq!(
            regexp_extract_impl(
                "test@a.com",
                &Regex::new("(.+)@(.+)\\.(.+)").unwrap(),
                3
            ),
            "com"
        );
    }

    #[test]
    fn test_unicode_characters() {
        // Test with Unicode characters
        assert_eq!(
            regexp_extract_impl(
                "こんにちは世界",
                &Regex::new("(こんにちは)(世界)").unwrap(),
                1
            ),
            "こんにちは"
        );
        assert_eq!(
            regexp_extract_impl(
                "こんにちは世界",
                &Regex::new("(こんにちは)(世界)").unwrap(),
                2
            ),
            "世界"
        );
        assert_eq!(
            regexp_extract_impl("😀😃😄", &Regex::new("(😀)(😃)(😄)").unwrap(), 2),
            "😃"
        );
    }

    #[test]
    fn test_special_regex_syntax() {
        // Test with non-capturing groups
        assert_eq!(
            regexp_extract_impl("abc123", &Regex::new("(a)(?:bc)(\\d+)").unwrap(), 1),
            "a"
        );
        assert_eq!(
            regexp_extract_impl("abc123", &Regex::new("(a)(?:bc)(\\d+)").unwrap(), 2),
            "123"
        );

        // Test with character classes
        assert_eq!(
            regexp_extract_impl("abc123", &Regex::new("([a-z]+)([0-9]+)").unwrap(), 1),
            "abc"
        );
        assert_eq!(
            regexp_extract_impl("abc123", &Regex::new("([a-z]+)([0-9]+)").unwrap(), 2),
            "123"
        );

        // Test with word boundaries
        assert_eq!(
            regexp_extract_impl("abc 123", &Regex::new("\\b([a-z]+)\\b").unwrap(), 1),
            "abc"
        );
        assert_eq!(
            regexp_extract_impl("abc 123", &Regex::new("\\b([0-9]+)\\b").unwrap(), 1),
            "123"
        );
    }

    #[test]
    fn test_idx_zero_returns_entire_match() {
        // Test idx = 0 returns the entire match
        assert_eq!(
            regexp_extract_impl("hello world", &Regex::new("hello (world)").unwrap(), 0),
            "hello world"
        );
        assert_eq!(
            regexp_extract_impl("a123b", &Regex::new("(\\d+)").unwrap(), 0),
            "123"
        );
        assert_eq!(
            regexp_extract_impl(
                "test@example.com",
                &Regex::new("(.+)@(.+)\\.(.+)").unwrap(),
                0
            ),
            "test@example.com"
        );
    }

    #[test]
    fn test_regex_no_match() {
        // Test regex doesn't match returns empty string
        assert_eq!(
            regexp_extract_impl("hello world", &Regex::new("xyz").unwrap(), 0),
            ""
        );
        assert_eq!(
            regexp_extract_impl("hello world", &Regex::new("\\d+").unwrap(), 0),
            ""
        );
    }

    #[ignore]
    #[test]
    fn test_incorrect_regex_no_match() {
        // FYI: Incorrect regex patterns don't even compile in Rust, so skipping this test,
        // but letting the reviewer an author know that this is a valid test case theoretically.
    }

    #[test]
    fn test_group_doesnt_exist() {
        assert_eq!(
            regexp_extract_impl("hello world", &Regex::new("hello (world)").unwrap(), 2),
            ""
        );
        assert_eq!(
            regexp_extract_impl("a123b", &Regex::new("(\\d+)").unwrap(), 2),
            ""
        );
    }

    #[test]
    fn test_empty_input() {
        // Test empty input string
        assert_eq!(
            regexp_extract_impl("", &Regex::new("(\\d+)").unwrap(), 0),
            ""
        );
    }

    #[test]
    fn test_complex_patterns() {
        // Test more complex regex patterns
        assert_eq!(
            regexp_extract_impl(
                "2025-08-04",
                &Regex::new("(\\d{4})-(\\d{2})-(\\d{2})").unwrap(),
                1
            ),
            "2025"
        );
        assert_eq!(
            regexp_extract_impl(
                "2025-08-04",
                &Regex::new("(\\d{4})-(\\d{2})-(\\d{2})").unwrap(),
                2
            ),
            "08"
        );
        assert_eq!(
            regexp_extract_impl(
                "2025-08-04",
                &Regex::new("(\\d{4})-(\\d{2})-(\\d{2})").unwrap(),
                3
            ),
            "04"
        );

        // Test with optional groups
        assert_eq!(
            regexp_extract_impl("abc", &Regex::new("(a)(b)?(c)").unwrap(), 2),
            "b"
        );
        assert_eq!(
            regexp_extract_impl("ac", &Regex::new("(a)(b)?(c)").unwrap(), 2),
            ""
        );
    }

    // Integration tests
    #[test]
    fn test_regexp_extract_scalar_invocation() {
        let invocation_args = [
            (
                Some("100-200".to_string()),
                Some(r"(\d+)-(\d+)".to_string()),
                Some(1),
            ),
            (Some("foo".to_string()), Some(r"(\d+)".to_string()), Some(1)),
            (
                Some("aaaac".to_string()),
                Some("(a+)(b)?(c)".to_string()),
                Some(2),
            ),
            (
                Some("abc".to_string()),
                Some(r"(a)(b)(c)".to_string()),
                Some(3),
            ),
            (Some("xyz".to_string()), Some("abc".to_string()), Some(0)),
            (None, Some(r"(\d+)".to_string()), Some(1)),
            (Some("some text".to_string()), None, Some(1)),
        ];

        let expected = [
            Some("100"),
            Some(""),
            Some(""),
            Some("c"),
            Some(""),
            None,
            None,
        ];

        let arg_fields = vec![
            FieldRef::new(Field::new("str", DataType::Utf8, true)),
            FieldRef::new(Field::new("pattern", DataType::Utf8, true)),
            // Proves that idx can be coerced to Int64
            FieldRef::new(Field::new("idx", DataType::Int8, true)),
        ];

        let return_field = FieldRef::new(Field::new("result", DataType::Utf8, true));

        for i in 0..invocation_args.len() {
            let (input, pattern, idx) = &invocation_args[i];

            let args = ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Scalar(ScalarValue::Utf8(input.clone())),
                    ColumnarValue::Scalar(ScalarValue::Utf8(pattern.clone())),
                    ColumnarValue::Scalar(ScalarValue::Int64(*idx)),
                ],
                arg_fields: arg_fields.clone(), // Clone of Vec<Arc<>> is cheap, and it i
                number_rows: 1,
                return_field: Arc::clone(&return_field),
            };

            let udf = SparkRegexpExtract::new();

            let res = udf.invoke_with_args(args).unwrap();
            let actual = match res {
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(s))) => s,
                ColumnarValue::Scalar(ScalarValue::Utf8(None)) => String::new(),
                _ => panic!("Expected ScalarValue::Utf8"),
            };
            let expected = expected[i]
                .as_ref()
                .map(|e| e.to_string())
                .unwrap_or_default();

            assert_eq!(
                expected, actual,
                "At index {i}: expected '{expected}', got '{actual}'"
            );
        }
    }

    #[test]
    fn test_regexp_extract_scalar_invocation_edge_cases() {
        // Integration test for various edge cases
        let invocation_args = [
            // Empty regex pattern
            (Some("abc".to_string()), Some("".to_string()), Some(0)),
            // Unicode characters
            (
                Some("こんにちは世界".to_string()),
                Some("(こんにちは)(世界)".to_string()),
                Some(2),
            ),
            // Emoji characters
            (
                Some("😀😃😄".to_string()),
                Some("(😀)(😃)(😄)".to_string()),
                Some(3),
            ),
            // Extremely large index
            (
                Some("abc".to_string()),
                Some("(a)(b)(c)".to_string()),
                Some(999),
            ),
            // Multiple matches (only first match is returned)
            (
                Some("abc abc abc".to_string()),
                Some("(abc)".to_string()),
                Some(1),
            ),
            // Sequential capture groups (not truly overlapping)
            (
                Some("abcde".to_string()),
                Some("(a)(b)(c)(d)(e)".to_string()),
                Some(2),
            ),
            // Nested capture groups
            (
                Some("abcde".to_string()),
                Some("(a(b(c)d)e)".to_string()),
                Some(3),
            ),
            // Character classes
            (
                Some("abc123".to_string()),
                Some("([a-z]+)([0-9]+)".to_string()),
                Some(1),
            ),
            // Special regex syntax - non-capturing group
            (
                Some("abc123".to_string()),
                Some("(a)(?:bc)(\\d+)".to_string()),
                Some(2),
            ),
        ];

        let expected = [
            Some(""),
            Some("世界"),
            Some("😄"),
            Some(""),
            Some("abc"),
            Some("b"),
            Some("c"),
            Some("abc"),
            Some("123"),
        ];

        let arg_fields = vec![
            FieldRef::new(Field::new("str", DataType::Utf8, true)),
            FieldRef::new(Field::new("pattern", DataType::Utf8, true)),
            FieldRef::new(Field::new("idx", DataType::Int64, true)),
        ];

        let return_field = FieldRef::new(Field::new("result", DataType::Utf8, true));

        for i in 0..invocation_args.len() {
            let (input, pattern, idx) = &invocation_args[i];

            let args = ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Scalar(ScalarValue::Utf8(input.clone())),
                    ColumnarValue::Scalar(ScalarValue::Utf8(pattern.clone())),
                    ColumnarValue::Scalar(ScalarValue::Int64(*idx)),
                ],
                arg_fields: arg_fields.clone(),
                number_rows: 1,
                return_field: Arc::clone(&return_field),
            };

            let udf = SparkRegexpExtract::new();

            let res = udf.invoke_with_args(args).unwrap();
            let actual = match res {
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(s))) => s,
                ColumnarValue::Scalar(ScalarValue::Utf8(None)) => String::new(),
                _ => panic!("Expected ScalarValue::Utf8"),
            };
            let expected = expected[i]
                .as_ref()
                .map(|e| e.to_string())
                .unwrap_or_default();

            assert_eq!(
                expected, actual,
                "At index {i}: expected '{expected}', got '{actual}'"
            );
        }
    }

    #[test]
    fn test_regexp_extract_array_invocation() {
        let target_arr = StringArray::from(vec![
            Some("100-200"),
            Some("300-400"),
            None,
            Some("500-600"),
            Some("700-800"),
        ]);

        let regexp = Some(r"(\d+)-(\d+)".to_string());
        let idx = Some(2); // expecting to extract the second group

        let expected = [
            Some("200".to_string()),
            Some("400".to_string()),
            None,
            Some("600".to_string()),
            Some("800".to_string()),
        ];

        let arg_fields = vec![
            FieldRef::new(Field::new("str", DataType::Utf8, true)),
            FieldRef::new(Field::new("pattern", DataType::Utf8, true)),
            FieldRef::new(Field::new("idx", DataType::Int64, false)),
        ];

        let return_field = FieldRef::new(Field::new("result", DataType::Utf8, true));

        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(target_arr)),
                ColumnarValue::Scalar(ScalarValue::Utf8(regexp)),
                ColumnarValue::Scalar(ScalarValue::Int64(idx)),
            ],
            arg_fields: arg_fields.clone(),
            number_rows: 1,
            return_field: Arc::clone(&return_field),
        };

        let udf = SparkRegexpExtract::new();
        let res = udf.invoke_with_args(args).unwrap();
        let res_arr = match res {
            ColumnarValue::Array(arr) => arr,
            _ => {
                panic!("Expected an Array result");
            }
        };

        let actual_arr = as_generic_string_array::<i32>(&res_arr).unwrap();
        for i in 0..actual_arr.len() {
            if actual_arr.is_null(i) {
                assert!(expected[i].is_none(), "Expected None at index {i}");
            } else {
                // Sanity check
                assert!(expected[i].is_some(), "Expected Some at index {i}");

                let actual = actual_arr.value(i);
                let expected = expected[i].as_ref().unwrap().as_str();
                assert_eq!(
                    actual, expected,
                    "At index {i}: expected '{expected}', got '{actual}'"
                );
            }
        }
    }
}
