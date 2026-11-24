# TabuMCP: the Tabularium memory server

TabuMCP gives coding agents a long-term memory they share across sessions. It runs as one local service and serves MCP over HTTP to every client on the machine.

## Requirements

- Node 22 or newer.
- [Ollama](https://ollama.com) running, with the embedding model you pick at install pulled. The installer offers to pull it.
- Linux (x64 or arm64) or macOS on Apple Silicon. On Windows, use WSL2.

## Install

```bash
npx tabumcp install
```

Then ask your agent to run the `tabumcp` skill. The install links the command as `~/.local/bin/tabumcp` and prints the `PATH` line to add if that folder is missing from it.

## Embedding model

The installer asks which local Ollama embedding model to use:

1. `nomic-embed-text`: recommended. Duplicate-catching is tuned for it.
2. `mxbai-embed-large`: works, but duplicate-catching is tuned for `nomic-embed-text`.
3. `bge-m3`: many languages. Works, but duplicate-catching is tuned for `nomic-embed-text`.
4. Any other Ollama embedding model: type its name.

For a scripted install, pass the model with `npx tabumcp install --embed-model <name>`. With no terminal and no flag, the installer uses `nomic-embed-text`.

The model is fixed after the first install. Re-running install keeps it, and the service refuses to open a store at a different vector width.

The full README covers profiles, scopes, the TypeSafe key and every command: https://github.com/simkaz12/tabularium#readme
