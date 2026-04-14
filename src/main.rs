use std::{
    env, fs,
    io::{self, Write},
    path::{Component, Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

use anyhow::{anyhow, bail, Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use wait_timeout::ChildExt;

const GEMINI_API_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
const DEFAULT_MODEL: &str = "gemini-2.5-flash";
const DEFAULT_REASONING_EFFORT: &str = "low";
const MAX_TOOL_ROUNDS: usize = 12;
const MAX_TOOL_TEXT: usize = 24_000;

#[derive(Debug)]
struct Args {
    model: String,
    reasoning_effort: String,
    no_color: bool,
    workspace_root: Option<PathBuf>,
}

#[derive(Clone)]
struct Harness {
    api_key: String,
    model: String,
    reasoning_effort: String,
    use_color: bool,
    workspace_root: PathBuf,
}

#[derive(Copy, Clone)]
enum StyleKind {
    Assistant,
    Dim,
    Error,
    Prompt,
    Status,
    Success,
    Title,
    Tool,
    ToolName,
    Value,
}

#[derive(Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatCompletionMessage,
}

#[derive(Deserialize)]
struct ChatCompletionMessage {
    content: Option<String>,
}

#[derive(Debug)]
struct ToolInvocation {
    name: String,
    args: Value,
}

#[derive(Serialize)]
struct ToolSpec {
    name: &'static str,
    signature: &'static str,
    description: &'static str,
}

#[derive(Deserialize)]
struct ReadFileArgs {
    #[serde(alias = "filename")]
    path: String,
}

#[derive(Deserialize)]
struct ListFilesArgs {
    #[serde(default = "default_dot_path")]
    path: String,
}

#[derive(Deserialize)]
struct EditFileArgs {
    path: String,
    #[serde(default, alias = "old_text")]
    old_str: String,
    #[serde(alias = "new_text")]
    new_str: String,
}

#[derive(Deserialize)]
struct BashArgs {
    #[serde(alias = "cmd")]
    command: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[derive(Serialize)]
struct FileEntry {
    filename: String,
    file_type: String,
    size_bytes: Option<u64>,
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let args = Args::parse()?;

    let api_key = env::var("GEMINI_API_KEY")
        .context("missing GEMINI_API_KEY in the environment or .env file")?;
    let workspace_root = determine_workspace_root(args.workspace_root.clone())?;

    let harness = Harness {
        api_key,
        model: args.model,
        reasoning_effort: args.reasoning_effort,
        use_color: !args.no_color,
        workspace_root,
    };

    harness.run_repl()
}

impl Args {
    fn parse() -> Result<Self> {
        let mut model = env::var("GEMINI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
        let mut reasoning_effort = env::var("GEMINI_REASONING_EFFORT")
            .unwrap_or_else(|_| DEFAULT_REASONING_EFFORT.to_string());
        let mut no_color = false;
        let mut workspace_root = env::var_os("WORKSPACE_ROOT").map(PathBuf::from);

        let mut argv = env::args().skip(1);
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--model" => {
                    model = argv.next().context("missing value for --model")?;
                }
                "--reasoning-effort" => {
                    reasoning_effort = argv
                        .next()
                        .context("missing value for --reasoning-effort")?;
                }
                "--workspace-root" => {
                    workspace_root = Some(PathBuf::from(
                        argv.next().context("missing value for --workspace-root")?,
                    ));
                }
                "--no-color" => no_color = true,
                "--help" | "-h" => {
                    print_cli_help();
                    std::process::exit(0);
                }
                "--version" | "-V" => {
                    println!("rust-harness {}", env!("CARGO_PKG_VERSION"));
                    std::process::exit(0);
                }
                other => bail!("unknown argument: {other}"),
            }
        }

        Ok(Self {
            model,
            reasoning_effort,
            no_color,
            workspace_root,
        })
    }
}

impl Harness {
    fn run_repl(&self) -> Result<()> {
        self.print_banner();

        let mut conversation = vec![ChatMessage {
            role: "system".to_string(),
            content: self.build_system_prompt(),
        }];

        loop {
            print!("{}", self.style("you > ", StyleKind::Prompt));
            io::stdout().flush().context("failed to flush stdout")?;

            let mut input = String::new();
            let bytes_read = io::stdin()
                .read_line(&mut input)
                .context("failed to read stdin")?;

            if bytes_read == 0 {
                println!();
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            match input {
                "/exit" | "/quit" => break,
                "/help" => {
                    self.print_help();
                    continue;
                }
                "/reset" => {
                    conversation.truncate(1);
                    self.status("reset", "conversation cleared");
                    continue;
                }
                _ => {}
            }

            conversation.push(ChatMessage {
                role: "user".to_string(),
                content: input.to_string(),
            });

            if let Err(error) = self.run_agent_loop(&mut conversation) {
                self.error(&error.to_string());
            }
        }

        Ok(())
    }

    fn run_agent_loop(&self, conversation: &mut Vec<ChatMessage>) -> Result<()> {
        for _ in 0..MAX_TOOL_ROUNDS {
            let assistant_response = self.execute_llm_call(conversation)?;
            let tool_invocations = extract_tool_invocations(&assistant_response);

            conversation.push(ChatMessage {
                role: "assistant".to_string(),
                content: assistant_response.clone(),
            });

            if tool_invocations.is_empty() {
                self.print_assistant_message(&assistant_response);
                return Ok(());
            }

            for invocation in tool_invocations {
                self.print_tool_call(&invocation);
                let tool_result = self.execute_tool(&invocation);
                self.print_tool_result(&tool_result);

                conversation.push(ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "tool_result({})",
                        serde_json::to_string(&tool_result)
                            .context("failed to serialize tool result")?
                    ),
                });
            }
        }

        bail!("agent stopped after hitting the max tool round limit ({MAX_TOOL_ROUNDS})")
    }

    fn execute_llm_call(&self, conversation: &[ChatMessage]) -> Result<String> {
        let spinner = spinner(&format!("thinking with {}", self.model));
        let request_body = json!({
            "model": self.model,
            "messages": conversation,
            "reasoning_effort": self.reasoning_effort,
        })
        .to_string();

        let output = Command::new("curl")
            .arg("-sS")
            .arg("-X")
            .arg("POST")
            .arg(GEMINI_API_URL)
            .arg("-H")
            .arg(format!("Authorization: Bearer {}", self.api_key))
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(&request_body)
            .arg("-w")
            .arg("\n__HTTP_STATUS__:%{http_code}")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("failed to invoke curl for Gemini API")?;
        spinner.finish_and_clear();

        if !output.status.success() {
            bail!(
                "curl failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let (body, status) = stdout
            .rsplit_once("\n__HTTP_STATUS__:")
            .ok_or_else(|| anyhow!("failed to parse HTTP status from curl output"))?;
        let status: u16 = status
            .trim()
            .parse()
            .context("failed to parse Gemini HTTP status code")?;

        if !(200..300).contains(&status) {
            bail!("Gemini API error ({status}): {}", body.trim());
        }

        let parsed: ChatCompletionResponse =
            serde_json::from_str(body).context("failed to parse Gemini API response JSON")?;

        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .filter(|content| !content.trim().is_empty())
            .ok_or_else(|| anyhow!("Gemini returned an empty assistant message"))
    }

    fn execute_tool(&self, invocation: &ToolInvocation) -> Value {
        let spinner = spinner(&format!("running {}", invocation.name));
        let result = match invocation.name.as_str() {
            "read_file" => self.read_file_tool(invocation.args.clone()),
            "list_files" => self.list_files_tool(invocation.args.clone()),
            "edit_file" => self.edit_file_tool(invocation.args.clone()),
            "bash" => self.bash_tool(invocation.args.clone()),
            other => Err(anyhow!("unknown tool '{other}'")),
        };
        spinner.finish_and_clear();

        match result {
            Ok(value) => value,
            Err(error) => json!({
                "ok": false,
                "tool": invocation.name,
                "error": error.to_string(),
            }),
        }
    }

    fn read_file_tool(&self, args: Value) -> Result<Value> {
        let args: ReadFileArgs =
            serde_json::from_value(args).context("invalid args for read_file")?;
        let full_path = self.resolve_workspace_path(&args.path)?;
        let content = String::from_utf8_lossy(
            &fs::read(&full_path)
                .with_context(|| format!("failed to read {}", full_path.display()))?,
        )
        .into_owned();

        Ok(json!({
            "ok": true,
            "tool": "read_file",
            "file_path": self.display_path(&full_path),
            "content": truncate_text(&content, MAX_TOOL_TEXT),
        }))
    }

    fn list_files_tool(&self, args: Value) -> Result<Value> {
        let args: ListFilesArgs =
            serde_json::from_value(args).context("invalid args for list_files")?;
        let full_path = self.resolve_workspace_path(&args.path)?;
        let mut entries = Vec::new();

        for entry in fs::read_dir(&full_path)
            .with_context(|| format!("failed to list {}", full_path.display()))?
        {
            let entry = entry?;
            let metadata = entry.metadata()?;
            entries.push(FileEntry {
                filename: entry.file_name().to_string_lossy().into_owned(),
                file_type: if metadata.is_dir() {
                    "dir".to_string()
                } else {
                    "file".to_string()
                },
                size_bytes: metadata.is_file().then_some(metadata.len()),
            });
        }

        entries.sort_by(|left, right| left.filename.cmp(&right.filename));

        Ok(json!({
            "ok": true,
            "tool": "list_files",
            "path": self.display_path(&full_path),
            "files": entries,
        }))
    }

    fn edit_file_tool(&self, args: Value) -> Result<Value> {
        let args: EditFileArgs =
            serde_json::from_value(args).context("invalid args for edit_file")?;
        let full_path = self.resolve_workspace_path(&args.path)?;

        if args.old_str.is_empty() {
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create parent directory {}", parent.display())
                })?;
            }

            fs::write(&full_path, args.new_str.as_bytes())
                .with_context(|| format!("failed to write {}", full_path.display()))?;

            return Ok(json!({
                "ok": true,
                "tool": "edit_file",
                "path": self.display_path(&full_path),
                "action": "created_or_overwritten",
            }));
        }

        let original = String::from_utf8_lossy(
            &fs::read(&full_path)
                .with_context(|| format!("failed to read {}", full_path.display()))?,
        )
        .into_owned();

        if !original.contains(&args.old_str) {
            return Ok(json!({
                "ok": false,
                "tool": "edit_file",
                "path": self.display_path(&full_path),
                "action": "old_str_not_found",
            }));
        }

        let edited = original.replacen(&args.old_str, &args.new_str, 1);
        fs::write(&full_path, edited.as_bytes())
            .with_context(|| format!("failed to write {}", full_path.display()))?;

        Ok(json!({
            "ok": true,
            "tool": "edit_file",
            "path": self.display_path(&full_path),
            "action": "edited",
        }))
    }

    fn bash_tool(&self, args: Value) -> Result<Value> {
        let args: BashArgs = serde_json::from_value(args).context("invalid args for bash")?;
        let timeout_secs = args.timeout_secs.unwrap_or(30).clamp(1, 300);
        let mut child = Command::new("bash")
            .arg("-lc")
            .arg(&args.command)
            .current_dir(&self.workspace_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to execute bash command")?;

        let output = match child
            .wait_timeout(Duration::from_secs(timeout_secs))
            .context("failed while waiting for bash command")?
        {
            Some(_) => child
                .wait_with_output()
                .context("failed to collect bash output")?,
            None => {
                child.kill().ok();
                child.wait().ok();
                bail!("command timed out after {timeout_secs} seconds");
            }
        };

        Ok(json!({
            "ok": output.status.success(),
            "tool": "bash",
            "command": args.command,
            "exit_code": output.status.code(),
            "stdout": truncate_text(&String::from_utf8_lossy(&output.stdout), MAX_TOOL_TEXT),
            "stderr": truncate_text(&String::from_utf8_lossy(&output.stderr), MAX_TOOL_TEXT),
        }))
    }

    fn build_system_prompt(&self) -> String {
        let tools = [
            ToolSpec {
                name: "read_file",
                signature: r#"read_file({"path":"src/main.rs"})"#,
                description:
                    "Read a UTF-8 text file from the workspace and return its full contents.",
            },
            ToolSpec {
                name: "list_files",
                signature: r#"list_files({"path":"."})"#,
                description:
                    "List the immediate files and directories inside a workspace directory.",
            },
            ToolSpec {
                name: "edit_file",
                signature: r#"edit_file({"path":"src/main.rs","old_str":"before","new_str":"after"})"#,
                description:
                    "Replace the first occurrence of old_str with new_str. If old_str is empty, create or overwrite the file with new_str.",
            },
            ToolSpec {
                name: "bash",
                signature: r#"bash({"command":"cargo test","timeout_secs":30})"#,
                description:
                    "Run a bash command in the workspace root and return exit code, stdout, and stderr.",
            },
        ];

        let tool_listing = tools
            .iter()
            .map(|tool| {
                format!(
                    "TOOL\n===\nName: {}\nDescription: {}\nSignature: {}\n===============\n",
                    tool.name, tool.description, tool.signature
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "You are a coding assistant inside a Rust CLI harness.\n\
             The workspace root is {}.\n\
             You cannot directly access the filesystem or shell. You must request tools.\n\
             Here are the tools you can execute:\n\n\
             {}\n\
             When you want to use a tool, reply with exactly one or more lines in this format and nothing else:\n\
             tool: TOOL_NAME({{JSON_ARGS}})\n\
             Use compact single-line JSON with double quotes.\n\
             After receiving a tool_result(...) message, continue the task.\n\
             If no tool is needed, respond normally.\n\
             Prefer reading files before editing them.\n\
             Do not invent tool results.\n\
             Keep answers concise.",
            self.workspace_root.display(),
            tool_listing
        )
    }

    fn resolve_workspace_path(&self, raw_path: &str) -> Result<PathBuf> {
        let raw_path = raw_path.trim();
        if raw_path.is_empty() {
            bail!("path cannot be empty");
        }

        let candidate = if Path::new(raw_path).is_absolute() {
            PathBuf::from(raw_path)
        } else {
            self.workspace_root.join(raw_path)
        };

        let normalized = normalize_path(&candidate);
        if !normalized.starts_with(&self.workspace_root) {
            bail!("path escapes workspace root");
        }

        Ok(normalized)
    }

    fn display_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.workspace_root)
            .unwrap_or(path)
            .display()
            .to_string()
    }

    fn print_banner(&self) {
        println!(
            "{} {}",
            self.style("Rust Harness", StyleKind::Title),
            self.style(&format!("({})", self.model), StyleKind::Dim)
        );
        println!(
            "{} {}",
            self.style("workspace", StyleKind::Dim),
            self.style(&self.workspace_root.display().to_string(), StyleKind::Value)
        );
        println!(
            "{}",
            self.style("Commands: /help, /reset, /exit", StyleKind::Dim)
        );
        println!();
    }

    fn print_help(&self) {
        println!("{}", self.style("Commands", StyleKind::Title));
        println!("  /help   show this help");
        println!("  /reset  clear the current conversation");
        println!("  /exit   quit the harness");
        println!();
    }

    fn print_assistant_message(&self, message: &str) {
        println!("{}", self.style("assistant", StyleKind::Assistant));
        println!("{}", message.trim());
        println!();
    }

    fn print_tool_call(&self, invocation: &ToolInvocation) {
        let args = truncate_text(&invocation.args.to_string(), 180);
        println!(
            "{} {} {}",
            self.style("tool", StyleKind::Tool),
            self.style(&invocation.name, StyleKind::ToolName),
            self.style(&args, StyleKind::Dim)
        );
    }

    fn print_tool_result(&self, result: &Value) {
        let ok = result.get("ok").and_then(Value::as_bool).unwrap_or(false);
        let prefix = if ok {
            self.style("result", StyleKind::Success)
        } else {
            self.style("result", StyleKind::Error)
        };
        println!(
            "{} {}",
            prefix,
            self.style(&summarize_tool_result(result), StyleKind::Dim)
        );
        println!();
    }

    fn status(&self, label: &str, message: &str) {
        println!(
            "{} {}",
            self.style(label, StyleKind::Status),
            self.style(message, StyleKind::Dim)
        );
    }

    fn error(&self, message: &str) {
        eprintln!("{} {}", self.style("error", StyleKind::Error), message);
        eprintln!();
    }

    fn style(&self, text: &str, kind: StyleKind) -> String {
        if !self.use_color {
            return text.to_string();
        }

        match kind {
            StyleKind::Assistant => format!("{}", text.bright_yellow().bold()),
            StyleKind::Dim => format!("{}", text.bright_black()),
            StyleKind::Error => format!("{}", text.bright_red().bold()),
            StyleKind::Prompt => format!("{}", text.bright_green().bold()),
            StyleKind::Status => format!("{}", text.bright_blue().bold()),
            StyleKind::Success => format!("{}", text.bright_green().bold()),
            StyleKind::Title => format!("{}", text.bright_cyan().bold()),
            StyleKind::Tool => format!("{}", text.bright_magenta().bold()),
            StyleKind::ToolName => format!("{}", text.bright_white().bold()),
            StyleKind::Value => format!("{}", text.bright_white()),
        }
    }
}

fn default_dot_path() -> String {
    ".".to_string()
}

fn extract_tool_invocations(text: &str) -> Vec<ToolInvocation> {
    let mut invocations = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if !line.starts_with("tool:") {
            continue;
        }

        let after_prefix = line["tool:".len()..].trim();
        let Some(open_paren) = after_prefix.find('(') else {
            continue;
        };

        if !after_prefix.ends_with(')') {
            continue;
        }

        let name = after_prefix[..open_paren].trim();
        let args_str = after_prefix[open_paren + 1..after_prefix.len() - 1].trim();

        let Ok(args) = serde_json::from_str::<Value>(args_str) else {
            continue;
        };

        invocations.push(ToolInvocation {
            name: name.to_string(),
            args,
        });
    }

    invocations
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }

    normalized
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let mut truncated = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            truncated.push_str("\n...[truncated]...");
            return truncated;
        }
        truncated.push(ch);
    }
    truncated
}

fn summarize_tool_result(result: &Value) -> String {
    let tool_name = result
        .get("tool")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    if !result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        return result
            .get("error")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("{tool_name} failed"));
    }

    match tool_name {
        "read_file" => {
            let path = result
                .get("file_path")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            format!("read {path}")
        }
        "list_files" => {
            let count = result
                .get("files")
                .and_then(Value::as_array)
                .map(|files| files.len())
                .unwrap_or(0);
            let path = result.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("listed {count} entries in {path}")
        }
        "edit_file" => {
            let path = result
                .get("path")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let action = result
                .get("action")
                .and_then(Value::as_str)
                .unwrap_or("updated");
            format!("{action} {path}")
        }
        "bash" => {
            let code = result
                .get("exit_code")
                .and_then(Value::as_i64)
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string());
            format!("bash exited with code {code}")
        }
        _ => format!("{tool_name} finished"),
    }
}

fn spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    spinner.set_message(message.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner
}

fn determine_workspace_root(explicit_root: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(path) = explicit_root {
        return canonicalize_workspace_root(path);
    }

    let current_dir = env::current_dir().context("failed to get current working directory")?;

    if let Some(root) = find_project_root(&current_dir) {
        return Ok(root);
    }

    canonicalize_workspace_root(current_dir)
}

fn canonicalize_workspace_root(path: PathBuf) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("failed to canonicalize workspace root {}", path.display()))
}

fn find_project_root(start: &Path) -> Option<PathBuf> {
    for candidate in start.ancestors() {
        if candidate.join(".git").exists() || candidate.join("Cargo.toml").exists() {
            if let Ok(canonical) = candidate.canonicalize() {
                return Some(canonical);
            }
        }
    }
    None
}

fn print_cli_help() {
    println!("rust-harness {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Usage:");
    println!(
        "  cargo run -- [--model MODEL] [--reasoning-effort LEVEL] [--workspace-root PATH] [--no-color]"
    );
    println!();
    println!("Options:");
    println!("  --model MODEL                Gemini model to use");
    println!("  --reasoning-effort LEVEL     Gemini reasoning effort");
    println!("  --workspace-root PATH        Override the detected workspace root");
    println!("  --no-color                   Disable ANSI colors");
    println!("  -h, --help                   Show this help");
    println!("  -V, --version                Show the version");
}
