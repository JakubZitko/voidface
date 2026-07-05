# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Voidface contributors

"""Every subcommand returns exit code 2 on a missing input file/dir.

Locks the R7.86-R7.88 UX improvements — no subcommand should
surface a raw FileNotFoundError traceback for the trivial "user
mistyped the path" case.
"""

from __future__ import annotations

from pathlib import Path


def test_train_missing_config(tmp_path: Path) -> None:
    from voidface_cli.main import main

    rc = main(["train", str(tmp_path / "does_not_exist.toml")])
    assert rc == 2


def test_export_missing_checkpoint(tmp_path: Path) -> None:
    from voidface_cli.main import main

    rc = main([
        "export", str(tmp_path / "ghost.pt"), str(tmp_path / "out.onnx"),
    ])
    assert rc == 2


def test_bench_missing_checkpoint(tmp_path: Path) -> None:
    from voidface_cli.main import main

    images = tmp_path / "images"
    images.mkdir()
    rc = main([
        "bench", str(tmp_path / "ghost.pt"), str(images),
    ])
    assert rc == 2


def test_bench_missing_images_dir(tmp_path: Path) -> None:
    from voidface_cli.main import main

    # Fake but existing checkpoint file so we reach the images check.
    ckpt = tmp_path / "ghost.pt"
    ckpt.write_bytes(b"not-a-real-torch-pickle")
    rc = main([
        "bench", str(ckpt), str(tmp_path / "no_such_dir"),
    ])
    # Either 2 (dir not found) is acceptable. The bad-pickle path
    # would raise before the images dir is checked, but we validated
    # the file exists first — so images check runs and returns 2.
    assert rc == 2


def test_package_missing_checkpoint(tmp_path: Path) -> None:
    from voidface_cli.main import main

    rc = main([
        "package", str(tmp_path / "ghost.pt"), str(tmp_path / "release"),
    ])
    assert rc == 2


def test_protect_video_missing_checkpoint(tmp_path: Path) -> None:
    from voidface_cli.main import main

    rc = main([
        "protect-video",
        str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"),
        "--use-generator", str(tmp_path / "ghost.pt"),
    ])
    assert rc == 2


def test_protect_video_missing_input(tmp_path: Path) -> None:
    from voidface_cli.main import main

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"not-a-real-torch-pickle")
    rc = main([
        "protect-video",
        str(tmp_path / "does_not_exist.mp4"), str(tmp_path / "out.mp4"),
        "--use-generator", str(ckpt),
    ])
    assert rc == 2


def test_report_missing_original(tmp_path: Path) -> None:
    from voidface_cli.main import main

    rc = main([
        "report",
        str(tmp_path / "ghost1.png"), str(tmp_path / "ghost2.png"),
    ])
    assert rc == 2
