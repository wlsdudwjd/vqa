# RoboVQA 데모 (`demo.py`) 동작 설명

이 문서는 `demo.py` 하나로 구현된 RoboVQA 인터랙티브 데모가 내부적으로 정확히 어떻게 동작하는지 설명합니다.

## 1. 이게 뭔가

로컬에 다운로드된 RoboVQA val 데이터셋(비디오+QA)에서 무작위 에피소드를 골라 보여주고, 그 에피소드에 대해 자유롭게 질문하면 로컬 GPU에서 돌아가는 실제 VLM(Qwen2.5-VL-7B-Instruct)이 답하는 웹 데모입니다. Gradio로 만든 웹 UI이며 브라우저에서 `http://127.0.0.1:7860`으로 접속합니다.

## 2. 왜 프로세스가 2개로 나뉘는가 (가장 중요한 설계 결정)

이 프로젝트를 만들면서 직접 겪은 문제: **`tensorflow`와 `torch`(+`transformers`)를 같은 파이썬 프로세스 안에서 같이 import하면 세그멘테이션 폴트(segfault)로 죽습니다.** `tensorflow-cpu`(GPU 미지원 빌드)로 바꿔도 동일하게 재현됐습니다 — 두 프레임워크가 각자 번들로 들고 있는 네이티브 CUDA/cuDNN 공유 라이브러리가 한 프로세스 안에서 충돌하는 것으로 보입니다.

그런데 이 데모는 둘 다 필요합니다:
- **TensorFlow**: 로컬 tfrecord 파일(RoboVQA 데이터셋 포맷)을 읽고 JPEG 프레임을 디코딩하는 데 필요
- **PyTorch/Transformers**: Qwen2.5-VL 모델을 GPU에서 돌리는 데 필요

그래서 **파일은 하나(`demo.py`)지만 실행되는 프로세스는 2개**로 분리했습니다:

```
$ python demo.py               → "메인 프로세스": Gradio 서버 + TensorFlow (데이터 로딩)
$ python demo.py --vlm-worker  → "워커 프로세스": PyTorch + Transformers (VLM 추론)
```

메인 프로세스가 시작할 때 **자기 자신을 `--vlm-worker` 인자로 서브프로세스로 재실행**합니다 (`launch_vlm_worker()` 함수, `subprocess.Popen([sys.executable, __file__, '--vlm-worker'], ...)`). 두 프로세스는 stdin/stdout 파이프로 JSON 메시지를 주고받으며 통신합니다. 이 구조 덕분에 코드는 한 파일에 다 있지만, 실제 `import torch`는 워커 프로세스에서만, `import tensorflow`는 메인 프로세스에서만 일어나서 충돌이 나지 않습니다.

`ps aux`로 보면 실제로 두 프로세스가 보입니다:
```
.../python demo.py
.../python demo.py --vlm-worker
```

## 3. 파일 구조 (함수별 역할)

`demo.py`는 위에서 아래로 4개 구역으로 나뉩니다.

### 3.1 `run_vlm_worker()` — 워커 프로세스의 메인 루프
- `torch`, `transformers`를 이 함수 **안에서만** import (모듈 최상단이 아님 — 메인 프로세스가 이 파일을 import할 때 torch가 로드되지 않도록 하기 위함)
- Qwen2.5-VL-7B-Instruct를 4bit 양자화(`bitsandbytes`, `load_in_4bit=True`)로 GPU(`cuda:0`)에 로드
- 로딩이 끝나면 stdout에 `READY` 한 줄을 출력 (메인 프로세스가 이 신호를 기다림)
- 이후 무한 루프: `sys.stdin`에서 한 줄씩(JSON) 읽어서 —
  1. base64로 인코딩된 이미지들을 디코딩해 PIL 이미지로 변환
  2. Qwen2.5-VL의 채팅 템플릿에 "이미지 여러 장 + 질문 텍스트"를 채워 프롬프트 구성
  3. `model.generate()`로 답변 생성 (`max_new_tokens` 만큼)
  4. `{"answer": "..."}` JSON을 stdout에 한 줄 출력

**요청 형식** (stdin, 한 줄):
```json
{"images": ["<base64 jpeg>", "<base64 jpeg>", ...], "question": "...", "max_new_tokens": 128}
```
**응답 형식** (stdout, 한 줄):
```json
{"answer": "..."}
```

### 3.2 `launch_vlm_worker()` / `ask_vlm()` — 메인 프로세스 쪽 클라이언트
- `launch_vlm_worker()`: 워커 서브프로세스를 띄우고 `READY`가 올 때까지 블로킹 대기. 반환값은 `subprocess.Popen` 객체(`vlm_proc`)이며, 이 객체를 데모가 살아있는 동안 계속 재사용합니다 (질문마다 모델을 새로 로드하지 않음 — 로딩에만 ~2분 걸리기 때문에 중요).
- `ask_vlm(proc, images, question, max_frames=8, max_new_tokens=128)`:
  1. 프레임이 `max_frames`(기본 8장)보다 많으면 `np.linspace`로 균등 간격 샘플링해서 줄임 (프롬프트가 너무 길어지지 않도록)
  2. 각 프레임을 JPEG로 인코딩 → base64 문자열로 변환
  3. JSON 요청을 만들어 `proc.stdin`에 쓰고 flush
  4. `proc.stdout`에서 응답 한 줄을 읽어 답변 문자열 반환

### 3.3 `Task` / `Tasks` / `fetch_question_answer()` — RoboVQA 텍스트 파싱
RoboVQA 데이터셋은 한 에피소드의 여러 QA 턴을 `<task:종류>...` 같은 태그로 구분해서 **하나의 긴 텍스트**에 다 이어붙여 저장합니다. 예:
```
<task:affordance:discriminative:discrete:False>place candy in the tray Q: possible right now? A: no
```
- `Tasks`: 이 긴 텍스트를 `<task:...>` 태그 기준으로 잘라서 `{태그: [텍스트, ...]}` 딕셔너리로 만듦
- `Task`: 그 안의 텍스트를 다시 `A:` 기준으로 잘라 (질문, 답변) 쌍을 만들고, 남아있는 `<...>` 태그를 제거
- `fetch_question_answer(text)`: 위 둘을 합쳐서 `(인덱스, 태그, 질문, 답변)` 리스트를 반환 — 데이터셋의 **정답** QA를 화면에 보여주는 데 씀

### 3.4 `run_gradio_app()` — 메인 프로세스의 진짜 앱
1. `tf.config.set_visible_devices([], 'GPU')`로 TensorFlow가 GPU를 아예 못 보게 막음 — 안 그러면 TF가 시작할 때 GPU 메모리의 상당 부분을 선점해버려서, 나중에 뜨는 VLM 워커가 쓸 VRAM이 부족해짐
2. `tfrecord/val/val*` 파일들을 전부 찾아서 **모든 에피소드를 raw bytes 리스트로 메모리에 로드** (`raw_records`, 약 1335개 에피소드/399MB — 매번 디스크에서 다시 읽지 않도록 시작할 때 한 번에 로드)
3. `launch_vlm_worker()`로 워커 프로세스를 띄움 (이 단계에서 ~2분 대기)
4. Gradio `Blocks` UI 구성:
   - **"랜덤 에피소드 불러오기" 버튼** → `load_random_episode()`
     - `raw_records`에서 무작위로 하나 골라 `parse_episode()`로 (프레임 리스트, 원본 텍스트) 파싱
     - 프레임들을 GIF로 저장(`/tmp/robovqa_demo_episode.gif`)해서 화면에 표시
     - `format_ground_truth()`로 정답 QA를 마크다운 텍스트로 변환해 표시
     - 프레임 리스트(numpy 배열들)는 `gr.State`에 저장 — 같은 세션 안에서 여러 질문을 이어서 할 수 있도록 서버 메모리에 유지됨
   - **자유 질문 입력창 + "질문하기" 버튼** → `answer_question()`
     - 현재 세션에 로드된 프레임이 있으면 `ask_vlm()`으로 워커에 질문을 보내고 답변을 받아 표시
     - 프레임이 아직 없으면(에피소드를 안 불러온 상태) 안내 메시지 표시
     - Enter 키(`question_box.submit`)로도 같은 동작을 하도록 연결돼 있음
5. `demo.launch()`로 서버 시작 (기본 포트 7860)

### 3.5 `main()` — 진입점
`argparse`로 `--vlm-worker` 플래그만 확인합니다. 이 플래그가 있으면 워커 루프(`run_vlm_worker`)를, 없으면 Gradio 앱(`run_gradio_app`)을 실행합니다. 이게 "파일 하나, 프로세스 두 개"를 가능하게 하는 스위치입니다.

## 4. 요청 하나가 처리되는 전체 흐름

사용자가 브라우저에서 질문을 입력하고 엔터를 누르면:

```
[브라우저] question_box.submit
    ↓
[메인 프로세스] answer_question(episode_images, question)
    ↓
[메인 프로세스] ask_vlm(vlm_proc, images, question)
    - 프레임 8장으로 서브샘플링
    - JPEG+base64 인코딩
    - JSON 한 줄을 vlm_proc.stdin에 write + flush
    ↓ (프로세스 경계, stdin 파이프)
[워커 프로세스] run_vlm_worker()의 for 루프가 그 줄을 읽음
    - 이미지 디코딩, 채팅 템플릿 구성
    - model.generate() (GPU에서 실제 추론, 여기가 제일 오래 걸림)
    - JSON 한 줄을 stdout에 print
    ↓ (프로세스 경계, stdout 파이프)
[메인 프로세스] ask_vlm()이 그 줄을 읽고 answer 필드를 반환
    ↓
[브라우저] answer_box에 답변 표시
```

## 5. 알아두면 좋은 동작 특성 / 제약

- **모델은 한 번만 로드됨**: 워커는 데모가 켜져 있는 동안 계속 살아있는 하나의 프로세스라, 질문마다 모델을 다시 로드하지 않습니다. 대신 데모를 처음 켤 때 ~2분 정도 로딩 시간이 걸립니다.
- **`max_new_tokens=128`**: 모델이 한 번에 생성할 수 있는 최대 토큰 수. 이보다 길게 답하려던 문장은 중간에 잘립니다. 값을 늘리면 안 잘리지만 응답이 느려집니다. (`run_vlm_worker`의 기본값과 `ask_vlm`의 기본값 두 군데 모두 있음)
- **`max_frames=8`**: 에피소드 프레임이 8장보다 많으면 균등 간격으로 8장만 뽑아서 모델에 보냅니다 (프롬프트 길이/속도 관리).
- **VRAM 사용량**: 4bit 양자화 덕분에 모델 자체는 약 6GB 정도만 사용 (16GB GPU 기준 여유 있음).
- **동시 사용자**: `vlm_proc`가 전역 변수 하나라, 여러 브라우저 탭에서 동시에 질문하면 질문들이 한 파이프를 놓고 순서가 꼬일 수 있습니다 — 지금은 "한 명이 순서대로 쓰는" 걸 전제로 한 간단한 구조입니다.
- **`gr.State`와 API 호출**: `episode_images`는 `gr.State`라 Gradio 내부적으로 브라우저 세션(session hash)별로 서버 메모리에 저장됩니다. HTTP API로 직접 호출할 때는 이 값이 입력 파라미터 목록에 나타나지 않고, 같은 클라이언트/세션으로 이어서 호출하면 자동으로 이전 상태를 이어받습니다.

## 6. 실행 / 종료

```bash
# 실행
cd "/home/jbnu/바탕화면/새 폴더"
.venv/bin/python demo.py
# → http://127.0.0.1:7860 를 브라우저에서 열기

# 종료 (메인 프로세스 + 워커 프로세스 둘 다 정리됨)
pkill -f demo.py
```

## 7. 확장하고 싶다면

- **다른 모델로 교체**: 파일 상단의 `VLM_MODEL_ID` 한 줄만 바꾸면 됩니다 (다른 Qwen2-VL/Qwen2.5-VL 계열 체크포인트로 교체 가능. 다른 아키텍처면 `run_vlm_worker()`의 `AutoProcessor`/`Qwen2_5_VLForConditionalGeneration` 부분도 맞춰 바꿔야 함).
- **train 스플릿도 브라우징**: 지금은 `tfrecord/val/`만 읽습니다. `run_gradio_app()`의 `filepaths = tf.io.gfile.glob(...)` 줄에서 `'val'`을 `'train'`으로 바꾸거나 두 경로를 합치면 됩니다.
- **답변 길이 조절**: `ask_vlm(...)` 호출 시 `max_new_tokens` 인자를 넘기면 됩니다 (현재는 UI에 노출 안 돼 있고 코드 기본값만 있음).
