#!/bin/bash
set -e

usage() {
    cat >&2 <<EOF
usage: DEPLOY_HOST=user@host DEPLOY_DIR=tabularium-v2 DEPLOY_HEALTH_URL=http://host:8789/v2/health ./deploy.sh

  DEPLOY_HOST        ssh target that runs the server
  DEPLOY_DIR         directory on that host to sync v2/ into (relative to the ssh home, or absolute)
  DEPLOY_HEALTH_URL  health endpoint to poll after the restart
  DEPLOY_UFW_CIDR    optional source range to open on port 8789 with ufw
EOF
    exit 2
}

HOST=${DEPLOY_HOST:-}
REMOTE_DIR=${DEPLOY_DIR:-}
HEALTH_URL=${DEPLOY_HEALTH_URL:-}
UFW_CIDR=${DEPLOY_UFW_CIDR:-}
[ -n "$HOST" ] && [ -n "$REMOTE_DIR" ] && [ -n "$HEALTH_URL" ] || usage

UNIT=tabularium-v2.service
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

rsync -az --delete \
    --exclude node_modules \
    --exclude '*.log' \
    --exclude '/eval' \
    "$HERE/" "$HOST:$REMOTE_DIR/"

REMOTE_ABS=$(ssh "$HOST" "cd $REMOTE_DIR && pwd")
REMOTE_USER=$(ssh "$HOST" id -un)

ssh "$HOST" "cd $REMOTE_DIR && npm ci --omit=dev"

RENDERED=$(mktemp)
sed -e "s|@DEPLOY_DIR@|$REMOTE_ABS|g" -e "s|@SERVICE_USER@|$REMOTE_USER|g" "$HERE/deploy/$UNIT" > "$RENDERED"
scp -q "$RENDERED" "$HOST:/tmp/$UNIT"
rm -f "$RENDERED"

UFW_RULE=true
if [ -n "$UFW_CIDR" ]; then
    UFW_RULE="sudo ufw allow from $UFW_CIDR to any port 8789 proto tcp comment 'tabularium-v2'"
fi

ssh "$HOST" "sudo install -m 644 /tmp/$UNIT /etc/systemd/system/$UNIT \
    && rm -f /tmp/$UNIT \
    && sudo systemctl daemon-reload \
    && sudo systemctl enable $UNIT \
    && $UFW_RULE \
    && sudo systemctl restart $UNIT \
    && systemctl is-active $UNIT"

for _ in $(seq 15); do
    curl -sf -m 5 "$HEALTH_URL" && break
    sleep 2
done

echo
if ! curl -sf -m 5 "$HEALTH_URL" > /dev/null; then
    echo "deploy failed: $HEALTH_URL did not answer after 15 attempts" >&2
    exit 1
fi
