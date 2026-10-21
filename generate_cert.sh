#!/usr/bin/env bash
# Creates a self-signed certificate in certs/ so the server starts in HTTPS/WSS mode. Browsers only allow the
# microphone on HTTPS pages (or on localhost), so this is what lets the web interface record from another machine.
#
#   ./generate_cert.sh                      # localhost, this machine's name and its IP addresses
#   ./generate_cert.sh 192.168.1.20 myhost  # add more names or addresses
#
# Browsers warn about a self-signed certificate once per address; accept it, or import certs/cert.pem as trusted.
set -euo pipefail

dir="${CERT_DIR:-certs}"
days="${CERT_DAYS:-825}"
mkdir -p "$dir"
if [ -e "$dir/cert.pem" ] && [ "${FORCE:-}" != "1" ]; then
    echo "$dir/cert.pem already exists (FORCE=1 to replace it)" >&2
    exit 1
fi

san="DNS:localhost,IP:127.0.0.1"
add() {
    case ",$san," in *",$1,"*) return ;; esac
    san="$san,$1"
}
host="$(hostname 2>/dev/null || true)"
[ -n "$host" ] && add "DNS:$host"
# This machine's IPv4 addresses, without container and VM bridges
for ip in $(ip -o -4 addr show scope global 2>/dev/null \
            | awk '$2 !~ /^(docker|br-|veth|virbr|cni|flannel|cali)/ {split($4, a, "/"); print a[1]}'); do
    add "IP:$ip"
done
for name in "$@"; do
    if [[ "$name" =~ ^[0-9.]+$ || "$name" == *:* ]]; then add "IP:$name"; else add "DNS:$name"; fi
done

openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days "$days" \
    -keyout "$dir/key.pem" -out "$dir/cert.pem" \
    -subj "/CN=${host:-speaker-diarization}" \
    -addext "subjectAltName=$san" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
    -addext "extendedKeyUsage=serverAuth" 2>/dev/null
# The container runs as uid 1000 and mounts certs/ read-only
chmod 644 "$dir/cert.pem"
chmod 600 "$dir/key.pem"
echo "Wrote $dir/cert.pem and $dir/key.pem for: ${san//,/ }"
