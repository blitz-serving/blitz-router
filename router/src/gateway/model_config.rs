// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// Auto-discovery of model configuration from HuggingFace config.json.
//
// blitz-router is a fork of TGI (text-generation-inference) but acts only as
// an inner cluster router, not a full inference system. Model-dependent params
// (max_total_tokens, max_input_length) should be auto-discovered from the
// model's config.json rather than requiring manual CLI specification.

use std::fs;
use std::path::Path;

/// Subset of HuggingFace model config relevant to the router.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Maximum context length the model supports.
    pub max_position_embeddings: usize,
    /// Model architecture type (e.g., "qwen3_moe", "llama").
    pub model_type: String,
}

/// Load model configuration from a HuggingFace `config.json` file.
///
/// Returns `None` if the file doesn't exist or can't be parsed,
/// with a warning logged.
pub fn load_model_config(model_dir: &Path) -> Option<ModelConfig> {
    let config_path = model_dir.join("config.json");
    let text = match fs::read_to_string(&config_path) {
        Ok(t) => t,
        Err(_) => {
            tracing::debug!("No config.json found at {}", config_path.display());
            return None;
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&text) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("Failed to parse {}: {e}", config_path.display());
            return None;
        }
    };

    let max_position_embeddings = json["max_position_embeddings"].as_u64().map(|v| v as usize)?;
    let model_type = json["model_type"].as_str().unwrap_or("unknown").to_string();

    Some(ModelConfig {
        max_position_embeddings,
        model_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_load_model_config_missing_dir() {
        assert!(load_model_config(Path::new("/nonexistent")).is_none());
    }

    #[test]
    fn test_load_model_config_valid() {
        let dir = tempfile::tempdir().unwrap();
        let config = r#"{
            "model_type": "qwen3_moe",
            "max_position_embeddings": 40960,
            "hidden_size": 2048
        }"#;
        let mut f = std::fs::File::create(dir.path().join("config.json")).unwrap();
        f.write_all(config.as_bytes()).unwrap();

        let result = load_model_config(dir.path());
        assert!(result.is_some());
        let mc = result.unwrap();
        assert_eq!(mc.max_position_embeddings, 40960);
        assert_eq!(mc.model_type, "qwen3_moe");
    }
}
