"""Конечная матрица существующих конфигураций; без подбора по test."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from main import LLMClient

ROOT = Path(__file__).resolve().parents[3]
GUIDED = "src/subcategory_extraction/google_decision/prompts/google_decision_guided.md"
BASELINE = "src/subcategory_extraction/google_decision/prompts/google_decision_baseline.md"


@dataclass(frozen=True)
class Configuration:
    """Одна опубликованная модель, совместимый промпт и фиксированные параметры."""

    key: str
    module: str
    environment: dict[str, str] = field(default_factory=dict)
    training_table: str | None = None


CONFIGURATIONS = (
    Configuration("regex", "regex"),
    Configuration("tfidf", "tfidf", {"TFIDF_ARTIFACT_PATH": "out/classifier-methods/tfidf.json"}, "case_train"),
    Configuration("tfidf::legacy", "tfidf", {"TFIDF_ARTIFACT_PATH": "out/tfidf.json"}),
    Configuration("learned_keywords", "learned_keywords", training_table="manual_train"),
    Configuration("semantic_encoder", "semantic_encoder"),
    Configuration("semantic_encoder::threshold-encoder", "semantic_encoder", {
        "SEMANTIC_ENCODER_ARTIFACT_PATH": "out/rubert-thresholds/encoder",
    }, "manual_train"),
    Configuration("semantic_encoder_calibrated", "semantic_encoder_calibrated", {
        "SEMANTIC_ENCODER_ARTIFACT_PATH": "out/rubert-thresholds/encoder",
    }, "manual_train"),
    Configuration("semantic_encoder_f1", "semantic_encoder_f1", training_table="manual_train"),
    Configuration("snippet_logistic_regression", "snippet_logistic_regression"),
    Configuration("semantic_evidence", "semantic_evidence"),
    Configuration("tinybert", "tinybert"),
    Configuration("local_jev", "local_jev", {"LOCAL_JEV_PROMPT": "english"}),
    Configuration("local_jev (guided_ru, threshold=0.5)", "local_jev", {"LOCAL_JEV_PROMPT": "guided_ru"}),
    Configuration("google_decision", "google_decision", {"GOOGLE_DECISION_PROMPT": "guided"}),
    Configuration("google_decision::baseline", "google_decision", {"GOOGLE_DECISION_PROMPT": "baseline"}),
    Configuration("jev", "jev", {"JEV_PROMPT_PATH": GUIDED}),
    Configuration("jev_calibrated", "jev_calibrated"),
    Configuration("jev::baseline", "jev", {"JEV_PROMPT_PATH": BASELINE}),
    Configuration("jev_boundaries", "jev_boundaries"),
    Configuration("jev_tfidf", "jev_tfidf", {"JEV_PROMPT_PATH": GUIDED}, "case_train"),
    Configuration("jev_tfidf::baseline", "jev_tfidf", {"JEV_PROMPT_PATH": BASELINE}, "case_train"),
    Configuration("jev_retrieval", "jev_retrieval", {"JEV_PROMPT_PATH": GUIDED}, "case_train"),
    Configuration("jev_retrieval::baseline", "jev_retrieval", {"JEV_PROMPT_PATH": BASELINE}, "case_train"),
    Configuration("evidence_jev", "evidence_jev"),
    Configuration("jev_arbiter", "jev_arbiter", {"JEV_PROMPT_PATH": GUIDED}, "case_train"),
    Configuration("jev_arbiter::baseline", "jev_arbiter", {"JEV_PROMPT_PATH": BASELINE}, "case_train"),
    Configuration("qwen_decision", "qwen_decision", {"QWEN_DECISION_PROMPT": "auto"}, "manual_train"),
    Configuration("qwen_decision::guided", "qwen_decision", {"QWEN_DECISION_PROMPT": "guided"}),
    Configuration("qwen_decision::boundaries", "qwen_decision", {"QWEN_DECISION_PROMPT": "boundaries"}),
)
BY_KEY = {config.key: config for config in CONFIGURATIONS}
LOCAL_METHODS = {"regex", "tfidf", "learned_keywords", "semantic_encoder",
                 "semantic_encoder_calibrated", "semantic_encoder_f1", "local_jev"}

# Значения фиксируются до оценки; .env не определяет скрытые варианты матрицы.
FIXED_ENVIRONMENT = {
    "JEV_MODEL": "jev-1.13.0", "LLM_MODEL_ID": "gemini-decision-1",
    "QWEN_DECISION_MODEL": "decision-model-preview", "STRUCTURED_LLM_MODEL": "gpt-6.1-terra",
    "JEV_THRESHOLD": "0.5", "GOOGLE_DECISION_THRESHOLD": "0.5",
    "JEV_TFIDF_WEIGHT": "0.5", "JEV_TFIDF_THRESHOLD": "0.5",
    "LOCAL_JEV_MODEL": "jevos-v5", "LOCAL_JEV_THRESHOLD": "0.5",
    "SEMANTIC_ENCODER_DEVICE": "cpu", "RUBERT_F1_DEVICE": "cpu",
    "TFIDF_ARTIFACT_PATH": "out/classifier-methods/tfidf.json",
    "LEARNED_KEYWORDS_ARTIFACT": "src/subcategory_extraction/learned_keywords/artifacts/learned-keywords.json",
    "SEMANTIC_ENCODER_ARTIFACT_PATH": "out/semantic-encoder",
    "RUBERT_F1_ARTIFACT_PATH": "src/subcategory_extraction/semantic_encoder_f1/artifacts",
    "JEV_REFERENCE_DATASET": "out/hybrid-training.jsonl",
    "JEV_REFERENCE_MAX_CHARS": "1800",
}


def configuration_metadata(config: Configuration) -> dict:
    """Сохранить настройки и SHA файлов модели/промпта без ключей доступа."""
    environment = FIXED_ENVIRONMENT | config.environment
    files = {}
    for name, value in environment.items():
        if not name.endswith(("_PATH", "_ARTIFACT", "_DATASET")):
            continue
        path = ROOT / value
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file():
                with candidate.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files[str(candidate.relative_to(ROOT))] = digest
    return {"key": config.key, "module": config.module, "environment": environment,
            "artifact_sha256": files, "training_table": config.training_table,
            "selection": "existing configurations fixed before test; no threshold search"}


def client_observations(pipeline: object) -> list[dict]:
    """Собрать usage и признаки кэша также у вложенных клиентов гибридов."""
    observations = []
    seen: set[int] = set()

    def visit(obj: object, name: str) -> None:
        """Обойти только явно известные связи пайплайнов, без обхода весов."""
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, LLMClient):
            observations.append({
                "client": name, "model": obj.model_id,
                "cache_hit": getattr(obj, "last_cache_hit", None),
                "http_attempts": getattr(obj, "last_attempts", None),
                "usage": getattr(obj, "last_usage", None),
            })
            return
        for attr in ("client", "extractor", "arbiter", "baseline", "local", "jev", "tfidf"):
            child = getattr(obj, attr, None)
            if child is not None:
                visit(child, f"{name}.{attr}")

    visit(pipeline, "pipeline")
    return observations


@contextmanager
def configured(config: Configuration, *, fresh: bool = False) -> Iterator[None]:
    """Изолировать настройки конфигурации и обходить кэш при замере latency."""
    environment = FIXED_ENVIRONMENT | config.environment
    previous = os.environ.copy()
    # Отдельные каталоги обходят кэш всех вложенных API-клиентов, не стирая его.
    cache_root = ROOT / "out/benchmark-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(dir=cache_root) as temporary:
            if fresh:
                for key in ("JEV_CACHE_DIR", "STRUCTURED_LLM_CACHE_DIR", "QWEN_DECISION_CACHE_DIR"):
                    environment[key] = str(Path(temporary) / key.lower())
                environment["QWEN_DECISION_CACHE"] = "0"
            for key, value in environment.items():
                if key.endswith(("_PATH", "_ARTIFACT", "_DATASET")):
                    value = str(ROOT / value)
                os.environ[key] = value
            yield
    finally:
        os.environ.clear()
        os.environ.update(previous)
