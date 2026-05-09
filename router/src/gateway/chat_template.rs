// Copyright 2025 Blitz-serving
// SPDX-License-Identifier: Apache-2.0
//
// Chat template rendering, decoupled from tokenization.
//
// Two modes:
//   - None:   inner-cluster router — prompts arrive pre-rendered, pass through as-is
//   - Python: public API gateway — embed Python via PyO3, use jinja2 for full compatibility
//
// Encoding (text → token IDs) is NOT done here — that stays in the Rust
// `tokenizers` crate via `TokenizerRender`.

use std::fs;
use std::path::Path;

use super::validation::ValidationError;
use crate::ChatMessage;

// ---------------------------------------------------------------------------
// load_chat_template — extract template string from tokenizer_config.json
// ---------------------------------------------------------------------------

/// Read the `chat_template` field from `tokenizer_config.json`.
///
/// Returns the raw Jinja2 template string, or an error message if the file
/// is missing, malformed, or lacks a `chat_template` field.
pub fn load_chat_template(tokenizer_dir: &Path) -> Result<String, String> {
    let config_path = tokenizer_dir.join("tokenizer_config.json");
    let text = fs::read_to_string(&config_path)
        .map_err(|e| format!("Cannot read {}: {e}", config_path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("Invalid JSON in {}: {e}", config_path.display()))?;

    // chat_template can be a string or an array of objects with "name" and "template" fields.
    // We handle the simple string case; array templates use the first entry.
    match &json["chat_template"] {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(arr) if !arr.is_empty() => {
            // Use the first template (or the one named "default" if present)
            for item in arr {
                if item.get("name").and_then(|n| n.as_str()) == Some("default") {
                    if let Some(tmpl) = item.get("template").and_then(|t| t.as_str()) {
                        return Ok(tmpl.to_string());
                    }
                }
            }
            // Fall back to first entry
            arr[0]
                .get("template")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
                .ok_or_else(|| {
                    format!("{}: chat_template array entry has no 'template' field", config_path.display())
                })
        }
        _ => Err(format!("{} has no chat_template field", config_path.display())),
    }
}

// ---------------------------------------------------------------------------
// ChatRenderer
// ---------------------------------------------------------------------------

/// Chat template renderer with two modes.
///
/// Each tokenizer worker gets its own clone. For the `Python` variant this
/// increments the `Py<PyAny>` reference count (cheap, GIL-guarded).
pub enum ChatRenderer {
    /// No rendering — input is already a fully-rendered prompt (inner-cluster mode).
    None,

    /// Embed Python via PyO3 — full jinja2 compatibility for any model.
    #[cfg(feature = "python-chat-template")]
    Python(PythonChatRenderer),
}

impl ChatRenderer {
    /// Render chat messages into a prompt string.
    ///
    /// - `None`: returns the concatenated message content as-is.
    /// - `Python`: acquires GIL, calls jinja2 template.render().
    pub fn render(&mut self, messages: &[ChatMessage]) -> Result<String, ValidationError> {
        match self {
            ChatRenderer::None => {
                // Pass through: caller sent a pre-rendered prompt.
                // messages[0].content contains the original input string.
                Ok(messages
                    .first()
                    .map(|m| m.content.clone())
                    .unwrap_or_default())
            }
            #[cfg(feature = "python-chat-template")]
            ChatRenderer::Python(renderer) => renderer.render(messages),
        }
    }

    /// Create a Python-backed renderer from a tokenizer directory path.
    #[cfg(feature = "python-chat-template")]
    pub fn python(tokenizer_dir: &Path) -> Result<Self, String> {
        Ok(ChatRenderer::Python(PythonChatRenderer::new(tokenizer_dir)?))
    }
}

impl Clone for ChatRenderer {
    fn clone(&self) -> Self {
        match self {
            ChatRenderer::None => ChatRenderer::None,
            #[cfg(feature = "python-chat-template")]
            ChatRenderer::Python(r) => {
                // Must hold the GIL to clone Py<PyAny> reference count
                pyo3::Python::with_gil(|_py| ChatRenderer::Python(r.clone()))
            }
        }
    }
}

impl std::fmt::Debug for ChatRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRenderer::None => write!(f, "ChatRenderer::None"),
            #[cfg(feature = "python-chat-template")]
            ChatRenderer::Python(_) => write!(f, "ChatRenderer::Python(...)"),
        }
    }
}

// ---------------------------------------------------------------------------
// PythonChatRenderer (PyO3 embedded Python)
// ---------------------------------------------------------------------------

#[cfg(feature = "python-chat-template")]
mod python_impl {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList, PyModule};

    /// Inline Python source that creates a jinja2 template renderer.
    ///
    /// `create_renderer(template_str)` compiles the template once and returns
    /// a callable `render(messages, add_generation_prompt) -> str`.
    const PYTHON_RENDERER_SOURCE: &str = r#"
from jinja2 import Environment, BaseLoader, TemplateError

def create_renderer(template_str):
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    # Some HF chat templates call raise_exception()
    env.globals["raise_exception"] = _raise_exception
    template = env.from_string(template_str)

    def render(messages, add_generation_prompt=True):
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    return render

def _raise_exception(msg):
    raise TemplateError(msg)
"#;

    /// Chat template renderer that embeds Python via PyO3.
    ///
    /// Stores a reference to a compiled Python callable. `Clone` increments
    /// the reference count (fast, GIL-guarded). Each tokenizer worker holds
    /// its own clone and acquires the GIL independently on each `render()` call.
    #[derive(Clone)]
    pub struct PythonChatRenderer {
        /// Python callable: `render(messages: list[dict], add_generation_prompt: bool) -> str`
        render_fn: Py<PyAny>,
    }

    impl PythonChatRenderer {
        /// Create a new renderer by loading the chat template from `tokenizer_config.json`
        /// and compiling it with Python's jinja2.
        pub fn new(tokenizer_dir: &Path) -> Result<Self, String> {
            let template_str = load_chat_template(tokenizer_dir)?;

            Python::with_gil(|py| {
                // Load our inline Python module
                #[allow(deprecated)]
                let module = PyModule::from_code_bound(
                    py,
                    PYTHON_RENDERER_SOURCE,
                    "chat_template_renderer",
                    "chat_template_renderer",
                )
                .map_err(|e| format!("Failed to load Python renderer module: {e}"))?;

                // Call create_renderer(template_str) to compile the template
                let create_renderer = module
                    .getattr("create_renderer")
                    .map_err(|e| format!("Failed to get create_renderer: {e}"))?;

                let render_fn = create_renderer
                    .call1((template_str,))
                    .map_err(|e| format!("Failed to compile chat template: {e}"))?;

                Ok(Self {
                    render_fn: render_fn.into(),
                })
            })
        }

        /// Render chat messages into a prompt string using the compiled jinja2 template.
        ///
        /// Acquires the GIL, converts messages to Python dicts, calls the render
        /// function, and extracts the result as a Rust String.
        pub fn render(&self, messages: &[ChatMessage]) -> Result<String, ValidationError> {
            Python::with_gil(|py| {
                // Convert &[ChatMessage] to Python list[dict]
                let py_messages: Vec<Bound<'_, PyDict>> = messages
                    .iter()
                    .map(|m| {
                        let dict = PyDict::new(py);
                        dict.set_item("role", &m.role).unwrap();
                        dict.set_item("content", &m.content).unwrap();
                        dict
                    })
                    .collect();
                let py_list = PyList::new(py, &py_messages)
                    .map_err(|e| ValidationError::Tokenizer(format!("Failed to create Python list: {e}")))?;

                // Call render(messages, add_generation_prompt=True)
                let kwargs = PyDict::new(py);
                kwargs
                    .set_item("add_generation_prompt", true)
                    .map_err(|e| ValidationError::Tokenizer(format!("Failed to set kwarg: {e}")))?;

                let result = self
                    .render_fn
                    .call(py, (py_list,), Some(&kwargs))
                    .map_err(|e| ValidationError::Tokenizer(format!("Chat template render failed: {e}")))?;

                result
                    .extract::<String>(py)
                    .map_err(|e| ValidationError::Tokenizer(format!("Failed to extract rendered string: {e}")))
            })
        }
    }
}

#[cfg(feature = "python-chat-template")]
pub use python_impl::PythonChatRenderer;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_renderer_none_passthrough() {
        let mut renderer = ChatRenderer::None;
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
        }];
        let result = renderer.render(&messages).unwrap();
        assert_eq!(result, "Hello, how are you?");
    }

    #[test]
    fn test_chat_renderer_none_empty() {
        let mut renderer = ChatRenderer::None;
        let messages: Vec<ChatMessage> = vec![];
        let result = renderer.render(&messages).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_load_chat_template_missing_file() {
        let result = load_chat_template(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    /// Test Python chat template rendering with real model tokenizer configs.
    /// Only runs on machines with models at /nvme/models/.
    #[cfg(feature = "python-chat-template")]
    #[test]
    fn test_python_chat_renderer() {
        let test_cases = [
            "Qwen2-7B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen3-8B",
        ];
        let model_cache = Path::new("/nvme/models");
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "What is deep learning?".to_string(),
        }];

        for model_name in test_cases {
            let local_path = model_cache.join(model_name);
            if !local_path.exists() || !local_path.is_dir() {
                eprintln!("Skipping {model_name}: model not found at {}", local_path.display());
                continue;
            }

            let mut renderer = ChatRenderer::python(&local_path)
                .unwrap_or_else(|e| panic!("Failed to create renderer for {model_name}: {e}"));

            let result = renderer.render(&messages);
            assert!(result.is_ok(), "Render failed for {model_name}: {:?}", result);
            let rendered = result.unwrap();
            println!("[{model_name}] Rendered:\n++++++++\n{rendered}\n--------");

            check_rendered_chat_message(model_name, &rendered, &messages[0].content);
        }
    }

    #[allow(unused)]
    fn check_rendered_chat_message(model_name: &str, message: &str, content: &str) {
        match model_name {
            "Qwen2-7B-Instruct" => {
                assert!(message.contains("<|im_start|>system"));
                assert!(message.contains("You are a helpful assistant."));
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            "Qwen2.5-7B-Instruct" => {
                assert!(message.contains("<|im_start|>system"));
                assert!(message.contains("You are Qwen, created by Alibaba Cloud."));
                assert!(message.contains("You are a helpful assistant."));
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            "Qwen3-8B" => {
                assert!(message.contains("<|im_start|>user"));
                assert!(message.contains(content));
                assert!(message.contains("<|im_start|>assistant"));
            }
            _ => panic!("Unknown model: {model_name}"),
        }
    }
}
