use std::io::Read;

pub(crate) const MAX_VALIDATION_INPUT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_TOOL_ID_BYTES: usize = 96;

pub(crate) fn read_validation_input(reader: impl Read) -> Result<String, String> {
    let mut bytes = Vec::new();
    reader
        .take(MAX_VALIDATION_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read stdin: {error}"))?;
    if bytes.len() as u64 > MAX_VALIDATION_INPUT_BYTES {
        return Err(format!(
            "stdin exceeds the {} byte validation limit",
            MAX_VALIDATION_INPUT_BYTES
        ));
    }
    String::from_utf8(bytes).map_err(|_| "stdin must be valid UTF-8 JSON".to_string())
}

pub(crate) fn validate_tool_id(tool: String) -> Result<String, String> {
    if tool.is_empty() {
        return Err("tool id must not be empty".to_string());
    }
    if tool.len() > MAX_TOOL_ID_BYTES {
        return Err(format!("tool id must be at most {MAX_TOOL_ID_BYTES} bytes"));
    }
    if !tool
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(
            "tool id may contain only ASCII letters, digits, '-', '_', and '.'".to_string(),
        );
    }
    Ok(tool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_input_accepts_valid_utf8() {
        assert_eq!(
            read_validation_input(br#"{"ok":true}"#.as_slice()).unwrap(),
            r#"{"ok":true}"#
        );
    }

    #[test]
    fn bounded_input_rejects_oversized_payload() {
        let input = vec![b'x'; MAX_VALIDATION_INPUT_BYTES as usize + 1];
        assert!(read_validation_input(input.as_slice())
            .unwrap_err()
            .contains("validation limit"));
    }

    #[test]
    fn tool_ids_are_bounded_ascii_tokens() {
        assert_eq!(
            validate_tool_id("cake_lpr-1.0".into()).unwrap(),
            "cake_lpr-1.0"
        );
        assert!(validate_tool_id("../solver".into()).is_err());
        assert!(validate_tool_id("".into()).is_err());
    }
}
