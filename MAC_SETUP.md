# Mac(Apple Silicon)에서 RoboVQA 데모 돌리기

이 문서는 `demo_mac.py`를 Mac에서 실행하기 위한 설정 가이드입니다. `demo.py`(CUDA/bitsandbytes 버전)와 구조는 동일하지만, VLM 추론 부분만 **MLX**(Apple Silicon 전용 프레임워크)로 바꾼 버전입니다.

> ⚠️ 이 파일과 `demo_mac.py`는 이 작업이 진행된 Linux 환경에서는 실행/검증할 수 없었습니다 (Mac이 없음). `mlx-vlm`의 정확한 API가 버전마다 조금씩 바뀌어왔기 때문에, 실행 중 에러가 나면 [mlx-vlm GitHub](https://github.com/Blaizzy/mlx-vlm)의 최신 사용법과 대조해서 같이 고쳐야 할 수 있습니다.

## 1. 필요한 것

- Apple Silicon Mac (M1/M2/M3/M4), 통합 메모리 16GB 이상
- Python 3.11 또는 3.12 (Homebrew: `brew install python@3.12`)
- 약 5GB 여유 디스크 (4bit 양자화된 모델 다운로드용)

## 2. 파일 옮기기

이 폴더에서 Mac으로 옮겨야 할 것:
- `demo_mac.py`
- `tfrecord/val/` 폴더 전체 (약 399MB) — AirDrop, USB, 클라우드 등으로 복사

또는, tfrecord를 옮기는 대신 Mac에서 직접 RoboVQA val split을 새로 받아도 됩니다:
```bash
# Mac에서
pip install huggingface_hub
huggingface-cli download Tianli/robovqa --repo-type dataset \
  --include "tfrecord/val/*" --local-dir ./robovqa_data
```
이 경우 `demo_mac.py`를 그 폴더 기준으로 실행하거나, 환경변수로 경로를 지정하세요:
```bash
export ROBOVQA_DATA_DIR=/path/to/robovqa_data
```

## 3. 파이썬 환경 구성

```bash
cd 옮긴_폴더
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install tensorflow          # Apple Silicon용 wheel이 자동으로 설치됨 (CPU만 사용)
pip install gradio pillow numpy
pip install mlx-vlm             # Apple MLX 기반 VLM 추론 라이브러리
```

`mlx-vlm`이 처음 실행될 때 `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` 모델을 Hugging Face에서 자동으로 다운로드합니다 (약 4~5GB, 인터넷 필요).

## 4. 실행

```bash
python demo_mac.py
```
콘솔에 `VLM worker ready.`가 뜨면 준비 완료, 브라우저에서 `http://127.0.0.1:7860` 접속.

첫 실행은 모델 다운로드 때문에 오래 걸릴 수 있습니다. 이후 실행부터는 캐시된 모델을 바로 씁니다 (`~/.cache/huggingface`).

## 5. 잘 안 될 때 체크리스트

- **`ModuleNotFoundError: mlx_vlm`**: `pip install mlx-vlm`이 제대로 됐는지, venv가 활성화됐는지 확인
- **`apply_chat_template`/`load_config` import 에러**: `mlx-vlm` 버전에 따라 함수 위치가 다를 수 있습니다. `python -c "import mlx_vlm; help(mlx_vlm)"` 또는 [공식 예제](https://github.com/Blaizzy/mlx-vlm#usage)를 참고해 `run_vlm_worker()` 안의 import/호출부만 맞춰 고치면 됩니다.
- **메모리 부족(swap 심함)**: 더 작은 모델로 교체 — `VLM_MODEL_ID`를 `mlx-community/Qwen2-VL-2B-Instruct-4bit` 등으로 변경
- **`tensorflow` import가 느리거나 경고가 많음**: 무시해도 됩니다 (GPU 안 쓰도록 이미 막아놨음). 거슬리면 `tensorflow-cpu` 대신 그냥 `tensorflow` 유지해도 무방 (Mac용은 어차피 CPU 최적화 wheel).

## 6. Linux(CUDA) 버전과의 차이 요약

| | `demo.py` (여기, Linux+NVIDIA) | `demo_mac.py` (Mac) |
|---|---|---|
| 추론 라이브러리 | `torch` + `transformers` | `mlx` + `mlx-vlm` |
| 양자화 | `bitsandbytes` 4bit (CUDA) | MLX 자체 4bit (사전 양자화된 체크포인트) |
| 모델 ID | `Qwen/Qwen2.5-VL-7B-Instruct` | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| 프로세스 분리 이유 | TF+torch 세그폴트 확인됨 | TF+MLX 충돌은 미확인이지만 안전하게 동일 구조 유지 |
| UI/데이터 로딩 코드 | 동일 | 동일 |
