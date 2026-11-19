---
name: tabumcp
description: Set up TabuMCP, the local memory server, right after `npx tabumcp install`. Use when the user says "set up tabumcp", "connect memory" or "tabumcp skill". Checks the install, sets up profiles, maps project folders to scopes, and connects Claude Code and Codex.
---

# tabumcp setup

Run these in order. Ask one question at a time and wait for the answer. Every command below uses the full path `~/.local/bin/tabumcp`, so it works whatever the user's `PATH` is.

## 1. Check the install

Run `~/.local/bin/tabumcp doctor`. If any line fails, show the failing lines to the user, say what each one needs (the line names the fix), and stop. Do not go on with a broken install.

## 2. Profiles

A profile is a memory area with its own shared scope. Repos that no rule maps fall into the `default` profile, which shares the `global` scope.

Ask whether the user wants work memory kept apart from the rest. If they say no, skip to step 4.

If yes, run `~/.local/bin/tabumcp profile add work`. One `work` profile is enough: map the work repos to it and leave the personal repos unmapped, so they stay in `default` and share `global`. The work profile gets its own shared scope, `global-work`, so a work record never shows up in a personal repo.

Then ask whether they also want a named `personal` profile. If so, run `~/.local/bin/tabumcp profile add personal` and map the personal repos to it in step 3 with `--profile personal`. A repo left unmapped still lands in `default`, not in `personal`.

## 3. Scopes

Ask which repos belong to work. A scope is one memory area. Repos that belong together, such as `acme-api` and `acme-web`, should share one scope.

For each group, run one command with a path pattern that matches every repo in the group:

```text
~/.local/bin/tabumcp scope add acme --profile work --path '~/Projects/acme-*'
```

Quote the pattern so the shell leaves the `*` alone. A repo with no scope uses its own folder name in the default profile, so unmapped repos need no command. If the user chose a `personal` profile, run the same command with `--profile personal` for each personal group. Check the result with `~/.local/bin/tabumcp scope list`.

## 4. Connect clients

Ask which clients to connect: Claude Code, Codex, or both. Then run `~/.local/bin/tabumcp connect claude` and `~/.local/bin/tabumcp connect codex` as they choose.

Each command registers the MCP server and adds a short instruction block to the client's own instruction file (`~/.claude/CLAUDE.md` or `~/.codex/AGENTS.md`). Tell the user the client needs a restart before it sees the new tools. For any other client, point them to the URL and token path that `~/.local/bin/tabumcp doctor` prints.

## 5. Optional TypeSafe key

Tell the user that a TypeSafe key switches on duplicate checks and the daily cleaner, and that memory works without one. If they have a key, ask them to run `~/.local/bin/tabumcp key set` in their own terminal and type the key at the hidden prompt.

Never ask for the key in this chat and never type it yourself. Do not read the env file either.

## 6. Test

The memory tools appear only after the client restarts, so you cannot call `memory_recall` in this session. Run `~/.local/bin/tabumcp doctor` as the check and report what it says.

Then ask the user to restart the client and try one recall in the new session: ask for something general such as "project setup". An empty result is fine on a new install. A tool error there means the connection failed, so rerun `~/.local/bin/tabumcp doctor`.

Finish by telling the user what you set up and which command undoes it: `~/.local/bin/tabumcp disconnect claude`, `~/.local/bin/tabumcp disconnect codex`, or `~/.local/bin/tabumcp uninstall`.
