# CLAUDE.md

드로잉 특화 레퍼런스 서비스(졸업 프로젝트)의 **기술 검증용 내부 웹앱**. 팀원 4~5명(비개발자 포함)이 각자 PC에서 로컬로 실행한다.
Windows, Python 3.14, GPU 없음(모든 기능이 CPU에서 동작해야 함).

## 실행

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt          # 팀원용
pip install -r requirements-dev.txt      # 개발자용 (테스트 도구 추가)
python -m playwright install chromium    # 개발자용, 브라우저 E2E 테스트
streamlit run app.py
```

- 단독 스크립트: `python training.py`(자동 태깅 학습), `python migrate_to_supabase.py`(로컬 데이터 → DB), `python check_supabase.py`(DB 연결 테스트)
- 포즈: `python predict_poses.py`(DWPose 미리 계산, 아직 없는 이미지만, 500장 약 25분), `python predict_poses.py --import x.json`(별도 환경 결과 가져오기), `python export_coco.py`(사람이 확정한 포즈 → COCO)
- MediaPipe는 팀 환경에 넣지 않는다(numpy·OpenCV 충돌). 비교용은 `tools/mediapipe_predict.py` 상단 안내대로 별도 가상환경에서 실행
- 테스트: `python -m pytest tests`
  - `tests/test_pose.py`: 매칭 단위 테스트 (DB·브라우저 불필요)
  - `tests/e2e/test_pin_editor.py`: 핀 편집기를 Chromium에서 직접 검증 (Streamlit·DB 불필요)
  - `tests/e2e/test_pose_app.py`, `test_pose_step2_app.py`, `test_pose_3d_app.py`: 실제 서버 + 브라우저 + Supabase 전체 흐름. 테이블이 없으면 자동으로 건너뜀
  - `tests/e2e/test_mannequin.py`: 마네킹을 Streamlit 없이 검증. 외부 네트워크 요청이 하나라도 있으면 실패(오프라인 보장)
  - Streamlit 화면 E2E에서 결과를 읽을 때는 고정 대기 대신 "바뀐 상태"가 화면에 나타나거나 사라질 때까지 기다린다 (재실행 전 DOM을 읽는 실수 방지)
  - 기존 페이지 확인은 `AppTest`로 **화면만 그려 볼 것**. 평가 버튼을 누르는 스모크 테스트는 실제 `eval_phash.csv`에 기록하므로 돌리지 않는다

## 구조

| 경로 | 역할 |
| --- | --- |
| `app.py` | `st.navigation` 진입점, 사이드바 "내 이름"(`st.session_state["evaluator"]`) |
| `views/` | 페이지 스크립트. 공통 UI는 `views/common.py` |
| `methods/` | 검색 방식 모듈. `SearchMethod` + 기능 믹스인(`ImageSearch`/`PairFinder`/`TagSource`), `METHOD_CLASSES`에 등록하면 해당 페이지에 자동 노출 |
| `core/dataset.py` | `metadata.csv` + `images/` 로딩(`read_metadata`는 순수 함수, `load_metadata`는 캐시 버전), 썸네일 |
| `core/db.py` | Supabase 클라이언트, `fetch_all`/`upsert`, `friendly_error`(한국어 오류) |
| `core/tagging.py` | 팀 태그(`tags`)·라벨(`labels`) DB 입출력, 예측 CSV |
| `core/pose.py` | 포즈 관절 정의·정규화·매칭 (순수 numpy, Streamlit/DB 의존 없음) |
| `core/poses.py` | `poses`·`pose_evals` DB 입출력 |
| `core/pose_models.py` | 사전학습 포즈 모델 → 13관절 변환(COCO·MediaPipe 매핑, 신뢰도 → 상태, 방향 추정) |
| `core/pose_predict.py` | 모델 추정 실행·저장 (웹 '바로 추정'·'미리 계산'과 CLI 공용) |
| `core/pose_accuracy.py` | 모델 vs 사람 정확도(관절별 오차, PCK@0.2, 검출률) — 순수 pandas |
| `web/` | **Streamlit에 의존하지 않는** 프론트엔드 모듈(순수 ES 모듈). MVP 화면으로 옮길 수 있게 유지 |
| `components/` | `web/` 모듈을 Streamlit에 붙이는 얇은 어댑터(`st.components.v2`) |
| `web/mannequin/` | 3D 마네킹(three.js를 생성자로 받음 → 번들러·로컬·CDN 어디서나). `demo.html`은 Streamlit 없이 쓰는 예시 |
| `static/vendor/three/` | three.js 0.186.1 원본(수정 금지). `.streamlit/config.toml`의 `enableStaticServing`으로 `/app/static/`에서 제공 → 오프라인 동작. **설정은 서버를 켤 때만 읽히므로, 바꾸면 앱을 다시 시작해야 한다** |
| `supabase/schema.sql` | DB 스키마 전체. 여러 번 실행해도 안전해야 함(`if not exists`, 정책 존재 확인) |
| `third_party/` | 프로젝트에 포함한 외부 코드와 라이선스 원문 |
| `drawing_ref_test/` | 로컬 데이터(이미지, 캐시, 모델). **Git 제외** |

## 규칙

- **화면 문구는 모두 한국어.** 오류도 팀원이 이해할 수 있는 한국어로(`db.friendly_error` 참고).
- CSV는 `utf-8-sig`로 쓰고 읽는다(엑셀 한글 깨짐 방지).
- 페이지 간 공유 상태는 `st.session_state`, 위젯 키 이름은 페이지 접두사를 붙인다(`label_`, `pose_` 등).
- `st.cache_resource`/`st.cache_data` 함수의 인자 이름을 `_`로 시작하면 캐시 키에서 빠진다. 모듈 리로드 대비는 `methods/__init__.py`의 클래스 id 키 참고.
- 무거운 계산 결과(Phash, 임베딩)는 `drawing_ref_test/cache/` 등에 파일로 캐시한다.
- `st.components.v2` 컴포넌트 키에는 `__`를 쓸 수 없다. 작성자 이름처럼 임의 문자가 들어가는 키는 해시로 바꾼다(`components/pose_editor.py`).
- v2 컴포넌트는 같은 키로 다시 그려질 때 JS 함수가 다시 호출된다. 편집 중 상태는 `data.version`이 같으면 유지하고, 바꾸려면 version을 바꾼다.
- 좌표는 이미지 크기에 대한 0~1 비율로 저장하고, 각도 계산 전에는 반드시 가로세로 비율을 보정한다.
- 포즈의 좌우는 **캐릭터 기준**(캐릭터의 왼팔 = `l_`). 화면에 안내 문구를 항상 표시한다.

## 데이터 안전 (중요)

- `SUPABASE.env`/`.env`의 키 값은 **절대 출력하지 않는다.** 키 종류 확인이 필요하면 접두사(`sb_secret_`/`sb_publishable_`)만 본다.
- `poses`의 `source='model'` 행(annotator = 모델 이름: `dwpose`, `mediapipe-heavy`)은 모델 결과다. 사람 데이터와 섞어 지우지 말 것. 사람 라벨은 `manual`/`model_corrected`.
- 테스트는 실제 팀 데이터를 건드리지 않는다. DB에 쓰는 테스트는 작성자/라벨러 이름 `__test__`를 쓰고 끝나면 그 행만 지운다.
- `drawing_ref_test/eval_phash.csv`, DB의 `labels`·`poses`·`pose_evals`는 팀이 직접 만든 데이터다. 덮어쓰기·삭제 전에 반드시 확인한다.
- secret 키(`service_role`)는 RLS를 무시한다. 팀원에게는 publishable 키를 나눠준다.

## DB (Supabase)

- 테이블: `images`, `tags`, `labels`, `predictions`, `poses`, `pose_evals`
- 스키마 변경 시 `supabase/schema.sql`에 추가하고, 사용자가 SQL Editor에서 실행해야 한다(API 키로는 DDL 불가).
- 새 테이블에는 GRANT(이 프로젝트는 자동 부여 안 됨)와 RLS 정책을 함께 추가한다.

## 라이선스

- 외부 코드를 포함하면 `third_party/<이름>/LICENSE`에 원문을 넣고 `THIRD_PARTY_NOTICES.md`에 기록한다.
- x6ud/pose-search(MIT)는 **아이디어만 참고**했다(코드 미포함). 3D 마네킹은 three.js로 직접 구현한다.
- 수집 이미지(Safebooru)는 팀 내부 검증용이며 저장소·외부에 재배포하지 않는다.

## 작업 방식

- 큰 기능은 명세서(`SPEC_*.md`) 기준으로 단계별 구현, 단계마다 커밋하고 멈춰서 보고한다.
- 기존 페이지 동작을 바꾸지 않는다. 바꿔야 하면 보고에 명시한다.
