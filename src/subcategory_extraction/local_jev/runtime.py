"""Загрузка проверенных файлов jevos и запуск собственного локального сервера."""

import atexit
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import monotonic, sleep
from typing import Callable
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = MODULE_DIR.parents[2] / "out/local-jev"
RELEASE_URL = "https://github.com/feder-cr/jev/releases/download/{release}/{filename}"
RUNTIME_ARCHIVE = "jev-linux-x64.tar.gz"
MODEL_ARCHIVE = "jevos-v5-openvino-int8.zip"
DOWNLOAD_TIMEOUT = 120
STARTUP_TIMEOUT = 60
logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    """Вычислить контрольную сумму без загрузки большого файла в память."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download(cache_dir: Path, filename: str, manifest: dict) -> Path:
    """Загрузить архив атомарно и сверить закреплённую контрольную сумму."""
    destination = cache_dir / filename
    expected = manifest["archives"][filename]
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    url = RELEASE_URL.format(release=manifest["release"], filename=filename)
    logger.info("Первая загрузка local_jev: %s", filename)
    with NamedTemporaryFile(dir=cache_dir, prefix=".download-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            stream.close()
            if _sha256(temporary) != expected:
                raise ValueError(f"Неверная SHA-256 скачанного архива {filename}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _model_valid(model_dir: Path, manifest: dict) -> bool:
    """Проверить размеры и хеши всех файлов модели, включая неполный кэш."""
    return all(
        (path := model_dir / name).is_file()
        and path.stat().st_size == info["bytes"]
        and _sha256(path) == info["sha256"]
        for name, info in manifest["files"].items()
    )


def _prepare_files(cache_dir: Path) -> tuple[Path, Path]:
    """Подготовить проверенную модель и официальный Linux runtime в локальном кэше."""
    manifest = json.loads((MODULE_DIR / "artifact-manifest.json").read_text(encoding="utf-8"))
    model_dir = cache_dir / "model"
    if not _model_valid(model_dir, manifest):
        archive = _download(cache_dir, MODEL_ARCHIVE, manifest)
        model_dir.mkdir(exist_ok=True)
        with ZipFile(archive) as zipped, TemporaryDirectory(dir=cache_dir) as temporary:
            staging = Path(temporary)
            for name, info in manifest["files"].items():
                member = zipped.getinfo(f"model/{name}")
                if member.file_size != info["bytes"]:
                    raise ValueError(f"Неверный размер файла модели {name}")
                with zipped.open(member) as source, (staging / name).open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            if not _model_valid(staging, manifest):
                raise ValueError("Контрольные суммы файлов модели local_jev не совпали")
            for name in manifest["files"]:
                (staging / name).replace(model_dir / name)
    runtime_dir = cache_dir / "runtime"
    executable = runtime_dir / "jev/jev"
    if not executable.is_file():
        archive = _download(cache_dir, RUNTIME_ARCHIVE, manifest)
        with TemporaryDirectory(dir=cache_dir) as temporary:
            with tarfile.open(archive) as packed:
                packed.extractall(temporary, filter="data")
            if not (Path(temporary) / "jev/jev").is_file():
                raise ValueError("Архив runtime не содержит исполняемый файл jev")
            if runtime_dir.exists():
                shutil.rmtree(runtime_dir)
            Path(temporary).replace(runtime_dir)
    return executable, model_dir


def _stop(process: subprocess.Popen) -> None:
    """Остановить только сервер, созданный текущим приложением."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def ensure_server(
    base_url: str, health_check: Callable[[], dict], api_key: str | None,
) -> dict:
    """При отсутствии сервера загрузить файлы и дождаться готовности дочернего процесса."""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError(
            "Автозагрузка runtime local_jev поддерживает Linux x86_64; "
            "на другой ОС запустите официальный сервер отдельно",
        )
    from fcntl import LOCK_EX, flock

    cache_dir = Path(os.getenv("LOCAL_JEV_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    threads = int(os.getenv("LOCAL_JEV_THREADS", "4"))
    if threads < 1:
        raise ValueError("LOCAL_JEV_THREADS должен быть положительным")
    with (cache_dir / ".startup.lock").open("a") as lock:
        flock(lock, LOCK_EX)
        try:
            return health_check()
        except URLError:
            pass
        executable, model_dir = _prepare_files(cache_dir)
        parsed = urlsplit(base_url)
        command = [
            str(executable), "serve", "--model-dir", str(model_dir),
            "--host", str(parsed.hostname), "--port", str(parsed.port or 80),
            "--threads", str(threads), "--warmup", "0", "--state-cache", "0", "--batch-tokens", "0",
        ]
        environment = os.environ.copy()
        environment.pop("JEV_API_KEY", None)
        if api_key:
            environment["JEV_API_KEY"] = api_key
        log_path = cache_dir / "server.log"
        logger.info("Запуск локального Jev на %s; журнал: %s", base_url, log_path)
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=environment,
            )
        deadline = monotonic() + STARTUP_TIMEOUT
        try:
            while monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Локальный Jev завершился при запуске; см. {log_path}")
                try:
                    health = health_check()
                    atexit.register(_stop, process)
                    return health
                except URLError:
                    sleep(0.1)
            raise TimeoutError(f"Локальный Jev не запустился за {STARTUP_TIMEOUT} с; см. {log_path}")
        except BaseException:
            _stop(process)
            raise
