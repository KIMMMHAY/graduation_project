"""포즈 모델 추정을 미리 계산해 DB(poses, source='model')에 저장한다.

  python predict_poses.py                      # DWPose로 아직 결과가 없는 이미지만 (500장 기준 약 25분)
  python predict_poses.py --limit 20           # 20장만
  python predict_poses.py --overwrite          # 이미 있는 결과도 다시 계산
  python predict_poses.py --import mp.json     # 다른 환경에서 만든 결과(JSON) 가져오기 (예: MediaPipe 비교용)

중간에 멈춰도 다시 실행하면 이어서 한다.
"""
import argparse
import sys
import time

from core.cli import quiet_streamlit, utf8_console

quiet_streamlit()

from core import db  # noqa: E402
from core.dataset import read_metadata  # noqa: E402
from core.pose_models import DEFAULT_MODEL, MODELS  # noqa: E402
from core.pose_predict import import_predictions, run_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="포즈 모델 추정 미리 계산")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--import", dest="import_path", help="tools/mediapipe_predict.py 등이 만든 JSON")
    ap.add_argument("--model-name", help="--import 시 DB에 저장할 모델 이름 (기본: JSON에 적힌 이름)")
    args = ap.parse_args()
    try:
        if args.import_path:
            r = import_predictions(args.import_path, args.model_name)
        else:
            print(f"{MODELS[args.model].label} 불러오는 중...", flush=True)
            model = MODELS[args.model]()
            t0 = time.perf_counter()

            def progress(n, total, image_id):
                eta = (time.perf_counter() - t0) / n * (total - n)
                print(f"\r{n}/{total}  {image_id}  남은 시간 약 {eta / 60:.0f}분   ", end="", flush=True)

            r = run_batch(read_metadata(), model, args.limit, args.overwrite, progress)
            print()
    except db.DBError as e:
        print(f"\n실패: {e}", file=sys.stderr)
        return 1
    print(f"완료: {r['model']} · 처리 {r['processed']}장 · 인물 검출 {r['detected']}장")
    return 0


if __name__ == "__main__":
    utf8_console()
    sys.exit(main())
