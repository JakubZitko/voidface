#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Enforce Rule 0: every source file begins with an SPDX identifier.
# Documentation is exempt. Vendored license texts are exempt.

set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

# Extensions we care about, and the comment marker they use.
scan() {
    local pattern="$1"
    while IFS= read -r -d '' file; do
        # Skip vendored license copies and empty placeholders.
        case "$file" in
            ./LICENSES/*) continue ;;
            ./COPYING)    continue ;;
        esac
        # An SPDX-License-Identifier line must appear in the first 5 lines.
        if ! head -5 "$file" | grep -q "SPDX-License-Identifier"; then
            echo "missing SPDX header: $file"
            fail=1
        fi
    done < <(find . -type f -name "$pattern" \
        -not -path "./.venv/*" \
        -not -path "./.git/*" \
        -not -path "./node_modules/*" \
        -not -path "./target/*" \
        -not -path "*/__pycache__/*" \
        -print0)
}

scan "*.py"
scan "*.pyi"
scan "*.sh"
scan "*.rs"
scan "*.ts"
scan "*.tsx"
scan "Makefile"

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "Add: '# SPDX-License-Identifier: MIT' (or the language equivalent)"
    echo "as the first non-shebang line of each listed file."
    exit 1
fi

echo "All source files carry an SPDX identifier."
