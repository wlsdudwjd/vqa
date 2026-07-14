"""FastAPI server: record a video on your phone, ask a question, get an
answer from a local Qwen2.5-VL-7B running on this machine's GPU.

This doesn't touch the RoboVQA dataset or tensorflow at all - it only reads
whatever video the phone uploads - so torch is imported directly in this
process (no subprocess isolation needed, unlike demo.py which also loads
tensorflow for tfrecord parsing).

Run:
    .venv/bin/python mobile_server.py
Then, with your phone on the same Wi-Fi as this machine, open in the phone's
browser:
    http://<이 컴퓨터의 LAN IP>:8000
(this machine's LAN IP can be found with `ip -4 addr show`)
"""

import base64
import os
import tempfile

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from transformers import (AutoProcessor, BitsAndBytesConfig,
                           Qwen2_5_VLForConditionalGeneration)

VLM_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
MAX_FRAMES = 8

app = FastAPI()
processor = None
model = None


@app.on_event('startup')
def load_model():
  global processor, model
  print('Loading VLM (this takes ~2 min)...', flush=True)
  processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      VLM_MODEL_ID,
      quantization_config=BitsAndBytesConfig(
          load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
      device_map='cuda:0',
  )
  print('VLM ready.', flush=True)


def extract_frames(video_path, max_frames=MAX_FRAMES):
  """Uniformly samples up to max_frames RGB frames from a video file."""
  cap = cv2.VideoCapture(video_path)
  total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frames = []
  if total > 0:
    idx = np.linspace(0, total - 1, min(max_frames, total)).astype(int)
    for i in idx:
      cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
      ok, frame = cap.read()
      if ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  else:
    # Some containers don't report a frame count; fall back to sequential read.
    all_frames = []
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      all_frames.append(frame)
    if all_frames:
      idx = np.linspace(0, len(all_frames) - 1, min(max_frames, len(all_frames))).astype(int)
      frames = [cv2.cvtColor(all_frames[i], cv2.COLOR_BGR2RGB) for i in idx]
  cap.release()
  return frames


def ask_vlm(frames, question, max_new_tokens=128):
  pil_images = [Image.fromarray(f) for f in frames]
  content = [{'type': 'image', 'image': img} for img in pil_images]
  content.append({'type': 'text', 'text': question})
  messages = [{'role': 'user', 'content': content}]
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True)
  inputs = processor(
      text=[text], images=pil_images, return_tensors='pt').to(model.device)
  with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
  answer = processor.batch_decode(
      output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
  return answer.strip()


@app.post('/ask')
async def ask(video: UploadFile, question: str = Form(...)):
  suffix = os.path.splitext(video.filename or '')[1] or '.mp4'
  with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(await video.read())
    tmp_path = tmp.name
  try:
    frames = extract_frames(tmp_path)
    if not frames:
      return JSONResponse(
          {'error': '영상에서 프레임을 추출하지 못했습니다.'}, status_code=400)
    answer = ask_vlm(frames, question)
    return {'answer': answer, 'num_frames': len(frames)}
  finally:
    os.remove(tmp_path)


@app.get('/', response_class=HTMLResponse)
def index():
  return '''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboVQA 모바일 데모</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 480px; margin: 0 auto; padding: 16px; background: #111; color: #eee; }
  h1 { font-size: 1.2rem; }
  input, textarea, button { width: 100%; box-sizing: border-box; padding: 12px; margin: 8px 0; border-radius: 8px; border: 1px solid #444; background: #222; color: #eee; font-size: 1rem; }
  button { background: #3b82f6; border: none; font-weight: bold; }
  button:disabled { background: #555; }
  video { width: 100%; border-radius: 8px; margin-top: 8px; }
  #answer { white-space: pre-wrap; padding: 12px; background: #1e1e1e; border-radius: 8px; min-height: 3em; }
  #status { color: #9ca3af; font-size: 0.9rem; }
</style>
</head>
<body>
  <h1>📹 RoboVQA 모바일 데모</h1>
  <p>영상을 찍고 질문을 입력하면 로컬 VLM이 답합니다.</p>

  <input type="file" id="videoInput" accept="video/*" capture="environment">
  <video id="preview" controls style="display:none"></video>

  <textarea id="question" placeholder="예: 지금 뭘 하고 있어?" rows="2"></textarea>
  <button id="submitBtn">질문하기</button>
  <p id="status"></p>
  <div id="answer"></div>

<script>
const videoInput = document.getElementById('videoInput');
const preview = document.getElementById('preview');
const submitBtn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const answerEl = document.getElementById('answer');

videoInput.addEventListener('change', () => {
  if (videoInput.files[0]) {
    preview.src = URL.createObjectURL(videoInput.files[0]);
    preview.style.display = 'block';
  }
});

submitBtn.addEventListener('click', async () => {
  const file = videoInput.files[0];
  const question = document.getElementById('question').value.trim();
  if (!file) { statusEl.textContent = '먼저 영상을 찍어주세요.'; return; }
  if (!question) { statusEl.textContent = '질문을 입력해주세요.'; return; }

  submitBtn.disabled = true;
  statusEl.textContent = '업로드 및 분석 중... (수 초~수십 초 걸릴 수 있어요)';
  answerEl.textContent = '';

  const formData = new FormData();
  formData.append('video', file);
  formData.append('question', question);

  try {
    const res = await fetch('/ask', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      statusEl.textContent = '오류: ' + data.error;
    } else {
      statusEl.textContent = `완료 (프레임 ${data.num_frames}장 사용)`;
      answerEl.textContent = data.answer;
    }
  } catch (e) {
    statusEl.textContent = '요청 실패: ' + e;
  } finally {
    submitBtn.disabled = false;
  }
});
</script>
</body>
</html>'''


if __name__ == '__main__':
  uvicorn.run(app, host='0.0.0.0', port=8000)
