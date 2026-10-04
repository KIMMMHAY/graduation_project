# 서드파티 고지 (Third-Party Notices)

이 프로젝트가 사용하거나 참고한 오픈소스와 라이선스입니다. 저장소에 코드를 포함한 경우 원문 라이선스를 `third_party/<이름>/LICENSE`에 둡니다.

## 참고만 한 프로젝트 (코드 미포함)

| 프로젝트 | 라이선스 | 참고한 내용 |
| --- | --- | --- |
| [x6ud/pose-search](https://github.com/x6ud/pose-search) — Copyright (c) 2021 x6ud | MIT | 포즈 매칭 아이디어: 뼈 방향 각도 오차 기반 점수, 허용 범위 컷오프, 좌우 반전 양방향 비교, 부위별 비교. 3D 마네킹을 드래그로 회전하는 조작 방식. **코드·3D 모델·사진 데이터는 가져오지 않았습니다.** 3D 모델(OBJ)은 출처 표기가 없어 사용하지 않습니다. |

## 저장소에 포함한 코드

(단계 3에서 three.js를 `third_party/three/`에 포함할 예정)

## Python 패키지 (pip로 설치, 저장소에 포함하지 않음)

| 패키지 | 버전 | 라이선스 |
| --- | --- | --- |
| streamlit | 1.56.0 | Apache-2.0 |
| pandas | 3.0.2 | BSD-3-Clause |
| numpy | 2.4.4 | BSD-3-Clause 외 |
| pillow | 12.2.0 | MIT-CMU |
| imagehash | 4.3.2 | BSD-2-Clause |
| pyyaml | 6.0.3 | MIT |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| joblib | 1.6.0 | BSD-3-Clause |
| open_clip_torch | 3.3.0 | MIT |
| torch | 2.14.1 | BSD-3-Clause 외 (Apache-2.0, MIT 등 포함) |
| torchvision | 0.29.1 | BSD-3-Clause |
| supabase | 2.32.0 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| rtmlib | 0.0.16 | Apache-2.0 |
| onnxruntime | 1.30.0 | MIT |
| opencv-python / opencv-contrib-python | 5.0.0.93 | Apache-2.0 |
| mediapipe (비교용, 별도 가상환경에서만) | 1.0.1 | Apache-2.0 |
| pytest (개발용) | 9.1.1 | MIT |
| playwright (개발용) | 1.63.0 | Apache-2.0 |

## 사전학습 모델 가중치 (실행 시 내려받음)

| 모델 | 출처 | 라이선스 |
| --- | --- | --- |
| CLIP ViT-B-32 `laion2b_s34b_b79k` | LAION / OpenCLIP (Hugging Face `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) | MIT |
| RTMW-dw-x-l (`rtmw-dw-x-l_simcc-cocktail14`, 관절 추정) | OpenMMLab RTMPose / DWPose (rtmlib가 내려받음) | Apache-2.0 |
| YOLOX-m Human-Art (`yolox_m_8xb8-300e_humanart`, 인물 검출) | OpenMMLab, Human-Art 데이터셋으로 학습 | 코드·가중치 Apache-2.0. **⚠️ 학습 데이터(Human-Art)의 이용 조건은 아직 확인하지 못함 — MVP 외부 공개 전 확인 필요** |
| MediaPipe Pose Landmarker heavy (비교용) | Google MediaPipe | Apache-2.0 |

## 데이터

- 검증용 이미지는 Safebooru에서 수집했으며 각 작가의 저작물입니다. 팀 내부 기술 검증에만 쓰고 저장소·외부에 재배포하지 않습니다(`drawing_ref_test/`는 Git 제외).
