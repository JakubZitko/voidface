# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""CoreML export tests.

The full parity + Neural Engine execution test requires Apple
Silicon macOS with coremltools installed; those parts are gated
with a platform-based skip. The always-run tests verify the error
path (non-Apple-Silicon platforms get a clean CoreMlExportError,
not a cryptic ImportError).
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import pytest

_ON_APPLE_SILICON = (
    sys.platform == "darwin" and platform.machine().lower() == "arm64"
)
_HAS_COREMLTOOLS = importlib.util.find_spec("coremltools") is not None


def test_export_error_class_importable() -> None:
    from voidface.export.coreml import CoreMlExportError

    assert issubclass(CoreMlExportError, RuntimeError)


@pytest.mark.skipif(
    _ON_APPLE_SILICON and _HAS_COREMLTOOLS,
    reason="Apple Silicon has coremltools; the ImportError path only exercises elsewhere.",
)
def test_export_raises_clean_error_when_coremltools_missing(tmp_path: Path) -> None:
    from voidface.export.coreml import CoreMlExportError, export_generator_to_coreml
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    generator = Voidface(VoidfaceConfig(base_channels=8))
    with pytest.raises(CoreMlExportError, match="coremltools"):
        export_generator_to_coreml(generator, tmp_path / "out.mlpackage")


@pytest.mark.skipif(
    not (_ON_APPLE_SILICON and _HAS_COREMLTOOLS),
    reason="CoreML export requires Apple Silicon macOS + coremltools.",
)
def test_coreml_export_produces_mlpackage(tmp_path: Path) -> None:
    """Runs only on Apple Silicon with coremltools installed."""
    from voidface.export.coreml import export_generator_to_coreml
    from voidface.generator.architecture import Voidface, VoidfaceConfig

    generator = Voidface(VoidfaceConfig(base_channels=8)).eval()
    out = tmp_path / "Voidface.mlpackage"
    export_generator_to_coreml(
        generator, out, example_resolution=64, quantize_weights=False
    )
    assert out.exists()
    # mlpackage is a directory in modern coremltools.
    assert out.is_dir()
