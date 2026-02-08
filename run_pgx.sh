#!/bin/bash
# Wrapper script to run python on PGX via Docker Exec
# Usage: ./run_pgx.sh script.py [args...]

CONTAINER_NAME="sionna-worker"

# Check if container is running
if ! ssh pgx-data "docker ps --format '{{.Names}}' | grep -q ^${CONTAINER_NAME}$"; then
    echo "❌ Container '${CONTAINER_NAME}' is not running on PGX."
    echo "   Please start the stack in Dockge first."
    exit 1
fi

# Ensure sionna is installed (simple check)
# Ideally this should be done in the image, but for now we check/install on fly if missing
# ssh pgx-data "docker exec ${CONTAINER_NAME} pip show sionna > /dev/null || docker exec ${CONTAINER_NAME} pip install sionna"

# Execute
# We pass the command to bash -c to handle arguments correctly
echo "🚀 Running on PGX (Docker): $@"
ssh -t pgx-data "docker exec -w /mnt/nas_data/sionna-sim ${CONTAINER_NAME} python $*"
