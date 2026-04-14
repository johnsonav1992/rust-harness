# rust-harness

A very small AI agent CLI harness in Rust.

It uses:

- Gemini via `GEMINI_API_KEY`
- a manual tool loop inspired by Mihail Eric's "The Emperor Has No Clothes"
- four local tools: `read_file`, `list_files`, `edit_file`, and `bash`

The model does not receive provider-native tool definitions. Instead, it gets a plain-text tool list in the system prompt and asks for tools with lines like:

```text
tool: read_file({"path":"src/main.rs"})
```

The harness parses those requests, runs the tool locally, prints colored tool activity, and sends back:

```text
tool_result({...})
```

## Requirements

- Rust toolchain
- `curl`
- a Gemini API key

## Setup

```bash
cp .env.example .env
```

Set `GEMINI_API_KEY` in `.env`.

## Run

```bash
cargo run --
```

Optional flags:

```bash
cargo run -- --model gemini-2.5-flash
cargo run -- --reasoning-effort low
cargo run -- --no-color
```

Inside the REPL:

- `/help` shows commands
- `/reset` clears conversation history
- `/exit` quits

## Notes

- `edit_file` replaces the first `old_str` match with `new_str`
- if `old_str` is empty, `edit_file` creates or overwrites the file
- file access is restricted to the current workspace root
- `bash` runs from the workspace root and supports `timeout_secs`
