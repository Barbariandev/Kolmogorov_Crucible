#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


TEST_SET_DIR="test_set_temp"
ARCHIVE_NAME="test.tar"
ENCRYPTED_ARCHIVE_NAME="test.tar.gpg"
TEST_KEY="${TEST_AES_HEX:-deadbeefcafebabe}"

NUM_ROWS=400
SKIP_ROWS=1000000

echo "→ Fetching test set data..."
rm -rf "$TEST_SET_DIR"
python3 "$SCRIPT_DIR/fetch_fineweb.py" --rows $NUM_ROWS --skip $SKIP_ROWS --out "$TEST_SET_DIR"

echo "→ Consolidating and archiving test set..."
mkdir -p "$TEST_SET_DIR/test"
cat "$TEST_SET_DIR"/*.txt > "$TEST_SET_DIR/test/test.txt"
tar -cvf test.tar -C "$TEST_SET_DIR/test" .

echo "→ Encrypting test set..."
gpg --batch --yes --passphrase "$TEST_KEY" --symmetric --cipher-algo AES256 test.tar

echo "→ Cleaning up..."
rm -r "$TEST_SET_DIR"
rm "$ARCHIVE_NAME"

echo "✓ Successfully created encrypted test set: $ENCRYPTED_ARCHIVE_NAME"
echo
echo "→ Next, place this file in the 'subnet' directory and build the image." 
