#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Fetch ensemble model weights into models/downloaded/.
# Individual URLs are populated as each surrogate lands.

set -euo pipefail

cd "$(dirname "$0")/.."

DEST="${VOIDFACE_MODEL_CACHE:-models/downloaded}"
mkdir -p "$DEST"

echo "Voidface model cache: $DEST"
echo ""
echo "This script is a scaffold. Individual model URLs are added as each"
echo "surrogate is implemented. See MAINTAINERS and Documentation/models/"
echo "for the list of surrogates."
