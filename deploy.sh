#!/bin/bash
set -e

usage() {
    cat >&2 <<EOF
usage: DEPLOY_HOST=user@host DEPLOY_DIR=path DEPLOY_URL=http://host:8787/mcp ./deploy.sh [--drain-only]

  DEPLOY_HOST  ssh target that runs the server
  DEPLOY_DIR   checkout directory on that host (relative to the ssh home, or absolute)
  DEPLOY_URL   MCP endpoint of the running server, used for the post-restart check
  DEPLOY_UV    optional uv binary on the host (default: uv)
  DEPLOY_UNIT  optional systemd unit name (default: tabularium-mcp.service)
EOF
    exit 2
}

HOST=${DEPLOY_HOST:-}
REMOTE_DIR=${DEPLOY_DIR:-}
MCP_URL=${DEPLOY_URL:-}
UV_BIN=${DEPLOY_UV:-uv}
UNIT=${DEPLOY_UNIT:-tabularium-mcp.service}
[ -n "$HOST" ] || usage
if [ "$1" != --drain-only ]; then
    [ -n "$REMOTE_DIR" ] && [ -n "$MCP_URL" ] || usage
fi
DRAIN_CAP=${DRAIN_CAP:-1800}
DRAIN_SETTLE=${DRAIN_SETTLE:-60}
DRAIN_INTERVAL=${DRAIN_INTERVAL:-5}

# Runs on the host. Prints pid and name of every claude/codex descendant of $1.
judges_under() {
    local procs
    procs=$(ps -eo pid=,ppid=,comm=) || return 1
    awk -v root="$1" '
        { ppid[$1] = $2; name[$1] = $3 }
        END {
            for (p in ppid) {
                for (q = p; q in ppid && q != root; q = ppid[q]);
                if (q == root && p != root && name[p] ~ /^(claude|codex)$/) print p, name[p]
            }
        }' <<< "$procs"
}

# Runs on the host. Cleaner passes run judge subprocesses under the service; a restart kills them mid-pass.
# Args: service pid, cap, settle window, poll interval (seconds). Never forces: on timeout it returns 1.
drain() {
    local pid=$1 cap=$2 settle=$3 interval=$4 start=$SECONDS idle_since="" now found
    [[ $pid =~ ^[0-9]+$ ]] || { echo "could not read service MainPID, not restarting" >&2; return 1; }
    [ "$pid" != 0 ] || { echo "service not running, nothing to drain"; return 0; }
    while :; do
        now=$SECONDS
        found=$(judges_under "$pid") || { echo "process listing failed, not restarting" >&2; return 1; }
        if [ -n "$found" ]; then
            idle_since=""
        else
            : "${idle_since:=$now}"
            [ $((now - idle_since)) -lt "$settle" ] || { echo "drained after $((now - start))s"; return 0; }
        fi
        if [ $((now - start)) -ge "$cap" ]; then
            echo "drain timed out after ${cap}s, not restarting. Still running:" >&2
            echo "${found:-(none right now, settle window incomplete)}" >&2
            return 1
        fi
        if [ -n "$found" ]; then
            echo "[$((now - start))s] waiting on $(echo "$found" | wc -l) judge process(es): $(echo "$found" | tr '\n' ' ')"
        else
            echo "[$((now - start))s] idle for $((now - idle_since))s of ${settle}s"
        fi
        sleep "$interval"
    done
}

# Runs on the host. Fails unless the PID is readable and no judge runs under it; PID 0 passes.
final_check() {
    local pid=$1 found
    [[ $pid =~ ^[0-9]+$ ]] || { echo "could not read service MainPID, not restarting" >&2; return 1; }
    [ "$pid" != 0 ] || return 0
    found=$(judges_under "$pid") || { echo "process listing failed, not restarting" >&2; return 1; }
    [ -z "$found" ] || { echo "judge started during the final check, not restarting: $found" >&2; return 1; }
}

remote_drain() {
    ssh "$HOST" "$(declare -f judges_under drain); drain \"\$(systemctl show -p MainPID --value $UNIT)\" $DRAIN_CAP $DRAIN_SETTLE $DRAIN_INTERVAL"
}

if [ "$1" = --drain-only ]; then
    remote_drain
    exit
fi

# ~/.local/bin is not on PATH for non-interactive ssh; set DEPLOY_UV to an absolute path if uv lives there.
ssh "$HOST" "cd $REMOTE_DIR \
    && git pull \
    && $UV_BIN sync --frozen"

# Wait for the Cleaner to drain right before the restart, so a running pass is not killed.
remote_drain

# The restart is the deploy: a pull alone leaves the old code running.
# One last judge check right before it, to keep the race window small.
ssh "$HOST" "$(declare -f judges_under final_check); \
    final_check \"\$(systemctl show -p MainPID --value $UNIT)\" \
    && sudo systemctl restart $UNIT \
    && systemctl is-active $UNIT"

# Wait for the socket — systemd reports active before the server binds.
for _ in $(seq 15); do
    curl -sf -o /dev/null -H "Accept: application/json, text/event-stream" -H "Content-Type: application/json" \
        -X POST "$MCP_URL" \
        -d '{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"deploy","version":"0"}}}' \
        && break
    sleep 2
done

# Verify the running process actually serves the new code.
SID=$(curl -sf -i -H "Accept: application/json, text/event-stream" -H "Content-Type: application/json" \
    -X POST "$MCP_URL" \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"deploy","version":"0"}}}' \
    | grep -i '^mcp-session-id:' | tr -d '\r' | awk '{print $2}')

curl -sf -H "Accept: application/json, text/event-stream" -H "Content-Type: application/json" \
    -H "mcp-session-id: $SID" -X POST "$MCP_URL" \
    -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' > /dev/null

curl -sf -H "Accept: application/json, text/event-stream" -H "Content-Type: application/json" \
    -H "mcp-session-id: $SID" -X POST "$MCP_URL" \
    -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
    | grep -o '"name":"[a-z_]*"' | sort -u
