"""The figures of a rundir (its plot/ folder): listing, safe paths, first-page thumbnails of PDFs."""
import hashlib
import subprocess
from pathlib import Path

SHOWN = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".csv", ".txt", ".html"}


def list_plots(rundir: Path) -> list[dict]:
    """[{path, size, mtime}] of the figures under plot/, path relative to it, sorted."""
    plot = rundir / "plot"
    if not plot.exists():
        return []
    files = [p for p in plot.rglob("*") if p.is_file() and p.suffix.lower() in SHOWN]
    return [{"path": str(p.relative_to(plot)), "size": p.stat().st_size, "mtime": p.stat().st_mtime}
            for p in sorted(files)]


def inside(base: Path, relative: str) -> Path:
    """base / relative, refused (ValueError) when it resolves outside base or does not exist."""
    path = (base / relative).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"no such file: {relative}")
    return path


def thumbnail(pdf: Path, cache_dir: Path, dpi: int = 40) -> Path:
    """PNG of the first page of `pdf`, made once by pdftoppm and kept in cache_dir (keyed by path and mtime)."""
    key = hashlib.sha1(f"{pdf}:{pdf.stat().st_mtime}:{dpi}".encode()).hexdigest()
    out = cache_dir / "thumbs" / f"{key}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", "1", "-singlefile", str(pdf),
                        str(out.with_suffix(""))], check=True, capture_output=True, timeout=60)
    return out
