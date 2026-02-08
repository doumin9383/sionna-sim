#!/bin/bash
# Script to mount NAS on PGX

echo "=== Mounting NAS on PGX ==="
echo "You will be asked for the password for 'sh-fukue' on PGX (for sudo)."

# 1. Create directory
ssh -t pgx-data "sudo mkdir -p /mnt/nas_data && sudo chown sh-fukue:sh-fukue /mnt/nas_data"

# 2. Add to fstab (if not exists)
MOUNT_ENTRY="192.168.100.1:/home/sh-fukue/Documents/Developments /mnt/nas_data nfs defaults,noatime,_netdev 0 0"
ssh -t pgx-data "grep -q '192.168.100.1' /etc/fstab || echo '$MOUNT_ENTRY' | sudo tee -a /etc/fstab"

# 3. Mount
ssh -t pgx-data "sudo mount -a"

# 4. Verify
ssh pgx-data "df -h | grep nas_data"
echo "=== Mount Complete ==="
