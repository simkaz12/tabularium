# TabuMCP: the Tabularium memory server

TabuMCP gives coding agents a long-term memory they share across sessions. It runs as one local service and serves MCP over HTTP to every client on the machine.

## Requirements

- Node 22 or newer.
- [Ollama](https://ollama.com) running, with the `nomic-embed-text` model pulled.
- Linux (x64 or arm64) or macOS on Apple Silicon. On Windows, use WSL2.

## Install

```bash
npx tabumcp install
```

Then ask your agent to run the `tabumcp` skill. The install links the command as `~/.local/bin/tabumcp` and prints the `PATH` line to add if that folder is missing from it.

The full README covers profiles, scopes, the TypeSafe key and every command: https://github.com/simkaz12/tabularium#readme
