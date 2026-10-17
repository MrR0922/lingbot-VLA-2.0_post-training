#!/usr/bin/env bash
set -Eeuo pipefail

: "${PASSWORD:?PlatforMAX must provide the PASSWORD environment variable}"
printf 'root:%s\n' "${PASSWORD}" | chpasswd
unset PASSWORD
exec /usr/sbin/sshd -D -e
