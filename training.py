"""CLIP 임베딩 + 태그별 로지스틱 회귀 학습.

웹앱의 '학습 실행' 페이지에서 쓰고, 단독으로도 실행할 수 있다:  python training.py
순서: 임베딩 추출(캐시에 없는 것만) → 태그별 학습(80/20 층화 분할) → 평가 → 전체 이미지 예측
"""
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from core.cli import quiet_streamlit, utf8_console

quiet_streamlit()

from core.dataset import read_metadata  # noqa: E402
from core.tagging import (EMB_IDS_CSV, EMB_NPY, MODELS_DIR, NEGATIVE, POSITIVE, PRED_CSV,  # noqa: E402
                          REPORT_CSV, TagDef, consolidate, load_labels, load_tags)

# 임베딩 모델 설정. WD Tagger 등으로 바꿀 때는 이 상수와 load_embedder / embed_images만 고치면 된다.
# EMBED_ID가 바뀌면 기존 임베딩 캐시는 자동으로 버리고 새로 추출한다.
EMBED_MODEL = "ViT-B-32"
EMBED_PRETRAINED = "laion2b_s34b_b79k"
EMBED_ID = f"open_clip/{EMBED_MODEL}/{EMBED_PRETRAINED}"

BATCH_SIZE = 32
MIN_POS = 20
MIN_NEG = 20
TEST_SIZE = 0.2
SEED = 42
F1_WARN = 0.7

Progress = Callable[[str, str, float | None], None]  # (단계, 메시지, 진행률 0~1 또는 None)


class TrainingError(Exception):
    def __init__(self, stage: str, target: str | None, cause: Exception):
        self.stage, self.target, self.cause = stage, target, cause
        where = f" ({target})" if target else ""
        super().__init__(f"'{stage}' 단계{where}에서 실패했습니다: {type(cause).__name__}: {cause}")


@dataclass
class TrainResult:
    status: pd.DataFrame                    # 태그별 라벨 현황
    report: pd.DataFrame | None = None      # 이번에 학습한 태그의 지표
    n_new_embeddings: int = 0
    seconds: float = 0.0
    log: list[str] = field(default_factory=list)

    @property
    def trained(self) -> bool:
        return self.report is not None and not self.report.empty


def _print_progress(stage: str, message: str, frac: float | None) -> None:
    pct = f" {frac:.0%}" if frac is not None else ""
    print(f"[{stage}]{pct} {message}", flush=True)


# ---------- 임베딩 ----------

def load_embedder():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(EMBED_MODEL, pretrained=EMBED_PRETRAINED, device="cpu")
    model.eval()
    return model, preprocess


def embed_images(embedder, paths: list[str]) -> np.ndarray:
    import torch
    model, preprocess = embedder
    batch = []
    for p in paths:
        with Image.open(p) as img:
            batch.append(preprocess(img.convert("RGB")))
    with torch.no_grad():
        feats = model.encode_image(torch.stack(batch))
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.numpy().astype(np.float32)


def load_embedding_cache() -> tuple[list[str], np.ndarray | None]:
    if not (EMB_NPY.exists() and EMB_IDS_CSV.exists()):
        return [], None
    ids = pd.read_csv(EMB_IDS_CSV, dtype=str, encoding="utf-8-sig")
    if ids.empty or (ids["model"] != EMBED_ID).any():
        return [], None  # 다른 모델로 만든 캐시는 쓰지 않는다
    vecs = np.load(EMB_NPY)
    if len(vecs) != len(ids):
        return [], None
    return list(ids["image_id"]), vecs


def _save_embedding_cache(ids: list[str], vecs: np.ndarray) -> None:
    np.save(EMB_NPY, vecs)
    pd.DataFrame({"image_id": ids, "model": EMBED_ID}).to_csv(EMB_IDS_CSV, index=False, encoding="utf-8-sig")


def ensure_embeddings(df: pd.DataFrame, get_embedder: Callable, progress: Progress) -> tuple[np.ndarray, int]:
    """df 행 순서대로 정렬된 임베딩 행렬과 새로 추출한 장수를 돌려준다."""
    ids, vecs = load_embedding_cache()
    have = set(ids)
    missing = [(i, p) for i, p in zip(df["id"], df["img_path"]) if i not in have]
    if missing:
        progress("임베딩 추출", f"모델 불러오는 중 ({EMBED_MODEL})", 0.0)
        embedder = get_embedder()
        for start in range(0, len(missing), BATCH_SIZE):
            chunk = missing[start:start + BATCH_SIZE]
            try:
                new = embed_images(embedder, [p for _, p in chunk])
            except Exception as e:
                raise TrainingError("임베딩 추출", f"이미지 {chunk[0][0]} ~ {chunk[-1][0]}", e) from e
            ids += [i for i, _ in chunk]
            vecs = new if vecs is None else np.vstack([vecs, new])
            _save_embedding_cache(ids, vecs)  # 배치마다 저장해서 중간에 끊겨도 이어서 할 수 있게
            done = start + len(chunk)
            progress("임베딩 추출", f"{done}/{len(missing)}장", done / len(missing))
    else:
        progress("임베딩 추출", f"캐시 사용 ({len(df)}장 모두 있음)", 1.0)
    row = {i: n for n, i in enumerate(ids)}
    return vecs[[row[i] for i in df["id"]]], len(missing)


# ---------- 라벨 현황 ----------

def label_status(tags: list[TagDef], consolidated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in tags:
        v = consolidated.loc[consolidated["tag_key"] == t.key, "value"]
        pos, neg = int((v == POSITIVE).sum()), int((v == NEGATIVE).sum())
        lacks = []
        if pos < MIN_POS:
            lacks.append(f"양성 {MIN_POS - pos}건")
        if neg < MIN_NEG:
            lacks.append(f"음성 {MIN_NEG - neg}건")
        rows.append({"tag_key": t.key, "name": t.name, "group": t.group, "n_pos": pos, "n_neg": neg,
                     "trainable": not lacks, "reason": "" if not lacks else "라벨 부족: " + ", ".join(lacks) + " 더 필요"})
    return pd.DataFrame(rows, columns=["tag_key", "name", "group", "n_pos", "n_neg", "trainable", "reason"])


# ---------- 학습 ----------

def _train_one(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    p, r, f1, _ = precision_recall_fscore_support(y_te, clf.predict(X_te), average="binary", zero_division=0)
    return clf, {"n_train": len(y_tr), "n_test": len(y_te), "precision": p, "recall": r, "f1": f1}


def train_all(get_embedder: Callable = load_embedder, progress: Progress = _print_progress) -> TrainResult:
    t0 = time.perf_counter()
    log: list[str] = []

    def report_progress(stage, message, frac=None):
        log.append(f"[{stage}] {message}")
        progress(stage, message, frac)

    try:
        df = read_metadata()
        tags = load_tags()
        labels = consolidate(load_labels())
        labels = labels[labels["image_id"].isin(set(df["id"]))]  # 다른 데이터셋에서 온 라벨은 무시
    except Exception as e:
        raise TrainingError("준비", None, e) from e

    status = label_status(tags, labels)
    targets = [t for t in tags if status.set_index("tag_key").at[t.key, "trainable"]]
    for _, s in status[~status["trainable"]].iterrows():
        report_progress("준비", f"{s['name']} 건너뜀 — {s['reason']}")
    if not targets:
        report_progress("준비", "학습 가능한 태그가 없습니다 (라벨 부족)")
        return TrainResult(status=status, seconds=time.perf_counter() - t0, log=log)

    X, n_new = ensure_embeddings(df, get_embedder, report_progress)
    row = {i: n for n, i in enumerate(df["id"])}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained_at = datetime.now().isoformat(timespec="seconds")
    reports, models = [], {}
    for n, t in enumerate(targets, start=1):
        report_progress("태그별 학습", f"{t.name} ({n}/{len(targets)})", (n - 1) / len(targets))
        try:
            data = labels[labels["tag_key"] == t.key]
            clf, metrics = _train_one(X[[row[i] for i in data["image_id"]]], data["value"].to_numpy())
            joblib.dump({"model": clf, "tag_key": t.key, "embed_id": EMBED_ID, "trained_at": trained_at},
                        MODELS_DIR / f"{t.key}.joblib")
        except Exception as e:
            raise TrainingError("태그별 학습", f"태그 {t.name}", e) from e
        models[t.key] = clf
        s = status.set_index("tag_key").loc[t.key]
        reports.append({"tag_key": t.key, "name": t.name, "group": t.group, "n_pos": s["n_pos"], "n_neg": s["n_neg"],
                        **metrics, "trained_at": trained_at, "embed_model": EMBED_ID})
    report = pd.DataFrame(reports)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    report_progress("평가", "지표 저장 완료 (train_report.csv)", 1.0)

    preds = []
    for n, (key, clf) in enumerate(models.items(), start=1):
        try:
            prob = clf.predict_proba(X)[:, list(clf.classes_).index(POSITIVE)]
        except Exception as e:
            raise TrainingError("전체 예측", f"태그 {key}", e) from e
        preds.append(pd.DataFrame({"image_id": df["id"], "tag_key": key, "prob": prob.round(4)}))
        report_progress("전체 예측", f"{n}/{len(models)} 태그", n / len(models))
    pd.concat(preds).to_csv(PRED_CSV, index=False, encoding="utf-8-sig")

    seconds = time.perf_counter() - t0
    report_progress("완료", f"{len(models)}개 태그 학습 · 새 임베딩 {n_new}장 · {seconds:.1f}초")
    return TrainResult(status=status, report=report, n_new_embeddings=n_new, seconds=seconds, log=log)


def main() -> int:
    try:
        result = train_all()
    except TrainingError as e:
        print(f"\n실패: {e}", file=sys.stderr)
        return 1
    print("\n라벨 현황")
    print(result.status[["name", "n_pos", "n_neg", "reason"]].to_string(index=False))
    if result.trained:
        print("\n평가 결과 (평가용 20%)")
        r = result.report[["name", "n_test", "precision", "recall", "f1"]].copy()
        r["주의"] = np.where(r["f1"] < F1_WARN, f"F1 < {F1_WARN}", "")
        print(r.to_string(index=False, float_format="%.2f"))
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
