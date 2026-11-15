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
GUIDED = "src/subcategory_extraction/decisions/jev/prompts/guided.md"
BASELINE = "src/subcategory_extraction/decisions/jev/prompts/baseline.md"
MORPHOLOGY_LOGREG = "lexical.morphology_logistic_regression"
ORDER_FORMAT_JOINT = "lexical.order_format_joint"
ORDER_FORMAT_WEIGHTS = "src/subcategory_extraction/lexical/order_format_joint/artifacts"
SNIPPET_WEIGHTS = {
    "SNIPPET_LOGISTIC_ARTIFACT": "src/subcategory_extraction/lexical/snippet_logistic_regression/artifacts/model.json.gz",
}


@dataclass(frozen=True)
class Configuration:
    """Одна опубликованная модель, совместимый промпт и фиксированные параметры."""

    key: str
    module: str
    environment: dict[str, str] = field(default_factory=dict)
    training_table: str | None = None


CONFIGURATIONS = (
    Configuration("regex", "lexical.regex"),
    Configuration("tfidf", "lexical.tfidf", {"TFIDF_ARTIFACT_PATH": "out/classifier-methods/tfidf.json"}, "case_train"),
    Configuration("tfidf::legacy", "lexical.tfidf", {"TFIDF_ARTIFACT_PATH": "out/tfidf.json"}),
    Configuration("learned_keywords", "lexical.learned_keywords", training_table="manual_train"),
    Configuration("semantic_encoder", "semantic.semantic_encoder"),
    Configuration("semantic_encoder::threshold-encoder", "semantic.semantic_encoder", {
        "SEMANTIC_ENCODER_ARTIFACT_PATH": "out/rubert-thresholds/encoder",
    }, "manual_train"),
    Configuration("semantic_encoder_calibrated", "semantic.semantic_encoder_calibrated", {
        "SEMANTIC_ENCODER_ARTIFACT_PATH": "out/rubert-thresholds/encoder",
    }, "manual_train"),
    Configuration("semantic_encoder_f1", "semantic.semantic_encoder_f1", training_table="manual_train"),
    Configuration("snippet_logistic_regression", "lexical.snippet_logistic_regression"),
    Configuration("morphology_logistic_regression", MORPHOLOGY_LOGREG, {
        "MORPH_LOGREG_MODEL": "avito-case-train",
    }),
    Configuration("morphology_logistic_regression::yandex-youla-decision1", MORPHOLOGY_LOGREG, {
        "MORPH_LOGREG_MODEL": "yandex-youla-decision1",
    }),
    Configuration("order_format_logistic_regression", "lexical.order_format_logistic_regression", SNIPPET_WEIGHTS),
    Configuration("order_format_phrase_logreg", "lexical.order_format_phrase_logreg", SNIPPET_WEIGHTS),
    Configuration("order_format_scope", "lexical.order_format_scope", SNIPPET_WEIGHTS),
    Configuration("order_format_veto", "lexical.order_format_veto", SNIPPET_WEIGHTS),
    Configuration("order_format_joint", ORDER_FORMAT_JOINT, SNIPPET_WEIGHTS),
    Configuration("order_format_joint::dedup", ORDER_FORMAT_JOINT, {
        "SNIPPET_LOGISTIC_ARTIFACT": f"{ORDER_FORMAT_WEIGHTS}/dedup.json.gz",
    }),
    Configuration("order_format_joint::yandex_youla", ORDER_FORMAT_JOINT, {
        "SNIPPET_LOGISTIC_ARTIFACT": f"{ORDER_FORMAT_WEIGHTS}/yandex_youla.json.gz",
    }),
    Configuration("semantic_evidence", "semantic.semantic_evidence"),
    Configuration("local_jev", "decisions.local_jev", {"LOCAL_JEV_PROMPT": "english"}),
    Configuration("local_jev (guided_ru, threshold=0.5)", "decisions.local_jev", {"LOCAL_JEV_PROMPT": "guided_ru"}),
    Configuration("google_decision", "decisions.google_decision", {"GOOGLE_DECISION_PROMPT": "guided"}),
    Configuration("google_decision::baseline", "decisions.google_decision", {"GOOGLE_DECISION_PROMPT": "baseline"}),
    Configuration("jev", "decisions.jev", {"JEV_PROMPT_PATH": GUIDED}),
    Configuration("jev_calibrated", "decisions.jev_calibrated"),
    Configuration("jev_semantic", "decisions.jev_semantic", {
        "JEV_PROVIDER": "selectel", "JEV_MODEL": "typesafe/jev-1.13",
        "JEV_ENDPOINT": "https://api.selectel.ru/aig/v1/decisions", "JEV_SEMANTIC_PROMPT": "auto",
    }, "manual_train"),
    Configuration("jev_semantic_arbiter", "hybrid.jev_semantic_arbiter", {
        "JEV_PROVIDER": "selectel", "JEV_MODEL": "typesafe/jev-1.13",
        "JEV_ENDPOINT": "https://api.selectel.ru/aig/v1/decisions", "JEV_SEMANTIC_PROMPT": "auto",
    }, "manual_train"),
    Configuration("jev::baseline", "decisions.jev", {"JEV_PROMPT_PATH": BASELINE}),
    Configuration("jev_boundaries", "decisions.jev_boundaries"),
    Configuration("jev_tfidf", "hybrid.jev_tfidf", {"JEV_PROMPT_PATH": GUIDED}, "case_train"),
    Configuration("jev_tfidf::baseline", "hybrid.jev_tfidf", {"JEV_PROMPT_PATH": BASELINE}, "case_train"),
    Configuration("jev_retrieval", "decisions.jev_retrieval", {"JEV_PROMPT_PATH": "src/subcategory_extraction/decisions/jev_retrieval/prompts/guided.md"}, "case_train"),
    Configuration("jev_retrieval::baseline", "decisions.jev_retrieval", {"JEV_PROMPT_PATH": "src/subcategory_extraction/decisions/jev_retrieval/prompts/baseline.md"}, "case_train"),
    Configuration("evidence_jev", "hybrid.evidence_jev"),
    Configuration("jev_arbiter", "hybrid.jev_arbiter", {"JEV_PROMPT_PATH": GUIDED}, "case_train"),
    Configuration("jev_arbiter::baseline", "hybrid.jev_arbiter", {"JEV_PROMPT_PATH": BASELINE}, "case_train"),
    Configuration("qwen_decision", "decisions.qwen_decision", {"QWEN_DECISION_PROMPT": "auto"}, "manual_train"),
    Configuration("qwen_decision::guided", "decisions.qwen_decision", {"QWEN_DECISION_PROMPT": "guided"}),
    Configuration("qwen_decision::boundaries", "decisions.qwen_decision", {"QWEN_DECISION_PROMPT": "boundaries"}),
)
BY_KEY = {config.key: config for config in CONFIGURATIONS}
LOCAL_METHODS = {"lexical.regex", "lexical.tfidf", "lexical.learned_keywords", "semantic.semantic_encoder",
                 "semantic.semantic_encoder_calibrated", "semantic.semantic_encoder_f1", "decisions.local_jev",
                 "lexical.snippet_logistic_regression", "semantic.semantic_evidence", MORPHOLOGY_LOGREG,
                 "lexical.order_format_logistic_regression", "lexical.order_format_phrase_logreg",
                 "lexical.order_format_scope", "lexical.order_format_veto", ORDER_FORMAT_JOINT}

# Значения фиксируются до оценки; .env не определяет скрытые варианты матрицы.
FIXED_ENVIRONMENT = {
    "JEV_MODEL": "jev-1.13.0", "LLM_MODEL_ID": "gemini-decision-1",
    "QWEN_DECISION_MODEL": "decision-model-preview", "STRUCTURED_LLM_MODEL": "gpt-6.1-terra",
    "JEV_THRESHOLD": "0.5", "GOOGLE_DECISION_THRESHOLD": "0.5",
    "JEV_TFIDF_WEIGHT": "0.5", "JEV_TFIDF_THRESHOLD": "0.5",
    "LOCAL_JEV_MODEL": "jevos-v5", "LOCAL_JEV_THRESHOLD": "0.5",
    "SEMANTIC_ENCODER_DEVICE": "cpu", "RUBERT_F1_DEVICE": "cpu",
    "TFIDF_ARTIFACT_PATH": "out/classifier-methods/tfidf.json",
    "LEARNED_KEYWORDS_ARTIFACT": "src/subcategory_extraction/lexical/learned_keywords/artifacts/learned-keywords.json",
    "SEMANTIC_ENCODER_ARTIFACT_PATH": "out/semantic-encoder",
    "RUBERT_F1_ARTIFACT_PATH": "src/subcategory_extraction/semantic/semantic_encoder_f1/artifacts",
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
    # Штатные веса и словари модуля тоже нужны, даже без *_ARTIFACT в окружении.
    module = ROOT / "src/subcategory_extraction" / config.module.replace(".", "/")
    for directory in (module / "artifacts", module / "dictionary"):
        for candidate in sorted(directory.rglob("*")):
            if candidate.is_file():
                with candidate.open("rb") as stream:
                    files[str(candidate.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
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
