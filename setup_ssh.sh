#!/bin/bash
# Helper script to copy SSH keys to PGX

echo "=== Copying SSH Key to PGX (LAN: 192.168.100.2) ==="
echo "You will be asked for the password for 'sh-fukue' on PGX."
ssh-copy-id -i ~/.ssh/id_ed25519.pub -o StrictHostKeyChecking=no pgx-data

echo ""
echo "=== Copying SSH Key to PGX (Tailscale: 100.89.160.90) ==="
echo "You will be asked for the password again."
ssh-copy-id -i ~/.ssh/id_ed25519.pub -o StrictHostKeyChecking=no pgx-control

echo ""
echo "=== Testing Connection ==="
ssh pgx-data "echo 'Success: Connected to PGX via LAN!'"
