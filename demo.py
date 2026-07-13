"""Standalone RoboVQA demo - everything in a single file.

Browse random episodes from the (locally downloaded) val split and ask a
real VLM free-form questions about what's happening in the video, alongside
the dataset's own ground-truth QA pairs for that episode.

Run with:
    .venv/bin/python demo.py
Then open the printed http://127.0.0.1:7860 URL in a browser.

Internally this file plays two roles depending on how it's invoked:
  - `python demo.py`              -> runs the Gradio app (imports tensorflow)
  - `python demo.py --vlm-worker` -> runs the VLM inference loop (imports
                                      torch/transformers)
These two roles MUST run in separate processes: importing torch/transformers
in the same process as tensorflow reliably segfaults here (confirmed even
with the CPU-only TF build - the two frameworks' bundled native CUDA/cuDNN
libraries conflict). The Gradio process therefore re-launches this same
file as a subprocess with --vlm-worker and talks to it over stdin/stdout
using newline-delimited JSON, keeping the model loaded for the whole
session instead of reloading it per request.
"""

import argparse
import base64
import io
import json
import os
import random
import re
import subprocess
import sys

DATA_DIR = os.environ.get('ROBOVQA_DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
TFRECORD_DIR = os.path.join(DATA_DIR, 'tfrecord')
GIF_PATH = '/tmp/robovqa_demo_episode.gif'
VLM_MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'


# =============================================================================
# VLM worker: runs in its own process, only ever imports torch/transformers.
# =============================================================================

def run_vlm_worker():
  import torch
  from PIL import Image
  from transformers import (AutoProcessor, BitsAndBytesConfig,
                             Qwen2_5_VLForConditionalGeneration)

  processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      VLM_MODEL_ID,
      quantization_config=BitsAndBytesConfig(
          load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
      device_map='cuda:0',
  )
  print('READY', flush=True)

  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue
    request = json.loads(line)
    pil_images = [
        Image.open(io.BytesIO(base64.b64decode(b))) for b in request['images']
    ]
    question = request['question']
    max_new_tokens = request.get('max_new_tokens', 128)

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
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True)[0].strip()

    print(json.dumps({'answer': answer}), flush=True)


# =============================================================================
# VLM client: used by the Gradio process to talk to the worker subprocess.
# =============================================================================

def launch_vlm_worker():
  """Starts this same file as a `--vlm-worker` subprocess, waits for READY."""
  proc = subprocess.Popen(
      [sys.executable, os.path.abspath(__file__), '--vlm-worker'],
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
      text=True, bufsize=1,
  )
  while True:
    line = proc.stdout.readline()
    if line.strip() == 'READY':
      break
    if not line:
      raise RuntimeError('vlm_worker died on startup:\n' + proc.stderr.read())
  return proc


def ask_vlm(proc, images, question, max_frames=8, max_new_tokens=128):
  """Answers a question given a list of HxWx3 uint8 frames, via the worker."""
  import numpy as np
  from PIL import Image

  if len(images) > max_frames:
    idx = np.linspace(0, len(images) - 1, max_frames).astype(int)
    images = [images[i] for i in idx]

  encoded = []
  for img in images:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='JPEG')
    encoded.append(base64.b64encode(buf.getvalue()).decode('ascii'))

  request = {
      'images': encoded,
      'question': question,
      'max_new_tokens': max_new_tokens,
  }
  proc.stdin.write(json.dumps(request) + '\n')
  proc.stdin.flush()

  response_line = proc.stdout.readline()
  if not response_line:
    raise RuntimeError('vlm_worker died:\n' + proc.stderr.read())
  return json.loads(response_line)['answer']


# =============================================================================
# RoboVQA text parsing: turns a raw episode text blob into (question, answer)
# pairs. RoboVQA packs many QA turns into one string using <task:...> tags.
# =============================================================================

class Task:
  """A class for handling tags and splits in a given task."""

  TAGS_RE = r'(</*\w[:\w]*>)'

  def __init__(self, text):
    self.text = text

  def get_splits(self, split_type='A:'):
    """Returns a list of (source, target) split pairs."""
    if split_type != 'A:':
      raise ValueError('Unknown split type: %s' % split_type)
    return self.get_splits_from_tags(start_tags=['A:'], end_tags=[])

  def get_splits_from_tags(self, start_tags, end_tags):
    """Returns a list of (source, target) split pairs given start/end tags."""
    split_positions = []
    position = 0
    while position < len(self.text):
      start_position = self.find_next_tag(position, start_tags)
      if start_position is None:
        break
      end_position = self.find_next_tag(start_position, end_tags)
      if end_position is None:
        end_position = len(self.text)
      split_positions.append((start_position, end_position))
      position = end_position + 1
    return self.get_splits_from_positions(split_positions)

  def get_splits_from_positions(self, split_positions):
    splits = []
    for (split_position, end_position) in split_positions:
      source = ''
      if split_position > 0:
        source = self._remove_tags(self.text[:split_position])
      target = self._remove_tags(self.text[split_position:end_position])
      splits.append((source, target))
    if not splits:
      splits = [('', self.text)]
    return splits

  def find_next_tag(self, position, tags):
    tag_position = None
    lower_text = self.text.lower()
    for tag in tags:
      p = lower_text.find(tag.lower(), position)
      if p >= 0 and (tag_position is None or p < tag_position):
        tag_position = p
    return tag_position

  def _remove_tags(self, text):
    return re.sub(self.TAGS_RE, '', text)


class Tasks:
  """A class for handling and holding tasks information."""

  TASK_RE = r'(<task[:\w]*>)'
  RE_FLAGS = re.IGNORECASE

  def __init__(self, tasks_raw):
    self.tasks_dict = {}
    self.add_from_text(tasks_raw)

  def add_from_text(self, text):
    split = re.split(self.TASK_RE, text, flags=self.RE_FLAGS)[1:]
    i = 0
    while i < len(split) - 1:
      tag = split[i].strip()
      task = split[i + 1].lstrip()
      if task:
        self.tasks_dict.setdefault(tag, []).append(task)
      i += 2


def fetch_question_answer(text):
  """Returns a list of (index, task_type, question, answer) for an episode."""
  tasks = Tasks(text)
  results = []
  for i, (task_type, task_list) in enumerate(tasks.tasks_dict.items()):
    for task in task_list:
      for question, answer in Task(task).get_splits('A:'):
        results.append((i, task_type, question.strip(), answer.strip()))
  return results


# =============================================================================
# Gradio app: runs in the main process, only ever imports tensorflow (for
# reading tfrecords) - never torch.
# =============================================================================

def run_gradio_app():
  import gradio as gr
  import tensorflow as tf
  from PIL import Image

  # TF is only used to read/decode tfrecords; keep it off the GPU so the VLM
  # worker subprocess has the full VRAM to itself.
  tf.config.set_visible_devices([], 'GPU')

  print('Loading val episodes into memory...')
  filepaths = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, 'val', 'val*'))
  raw_records = list(tf.data.TFRecordDataset(filepaths).as_numpy_iterator())
  print(f'Loaded {len(raw_records)} episodes.')

  print('Launching VLM worker (this loads the model, ~2 min)...')
  vlm_proc = launch_vlm_worker()
  print('VLM worker ready.')

  def parse_episode(raw_record):
    example = tf.train.SequenceExample()
    example.ParseFromString(raw_record)
    images = [
        tf.image.decode_jpeg(bl.bytes_list.value[0]).numpy()
        for bl in example.feature_lists.feature_list.get('images').feature
    ]
    text = example.feature_lists.feature_list.get(
        'texts').feature[0].bytes_list.value[0].decode('utf-8')
    return images, text

  def format_ground_truth(text):
    qa_list = fetch_question_answer(text)
    lines = [
        f'`{task_type.strip("<>")}`  \n{question} → {answer}'
        for _, task_type, question, answer in qa_list
    ]
    return '\n\n'.join(lines) if lines else '(no ground-truth QA pairs)'

  def load_random_episode():
    images, text = parse_episode(random.choice(raw_records))
    frames = [Image.fromarray(x) for x in images]
    frames[0].save(GIF_PATH, save_all=True, append_images=frames[1:],
                    duration=150, loop=0)
    return GIF_PATH, format_ground_truth(text), images, '', ''

  def answer_question(images, question):
    if not images:
      return '먼저 "랜덤 에피소드" 버튼을 눌러 영상을 불러오세요.'
    if not question.strip():
      return ''
    return ask_vlm(vlm_proc, images, question.strip())

  with gr.Blocks(title='RoboVQA Demo') as demo:
    gr.Markdown(
        '# RoboVQA 데모\n'
        '랜덤 에피소드를 불러온 뒤, 데이터셋의 정답 QA를 참고하거나 '
        '직접 자유롭게 질문해서 로컬 VLM(Qwen2.5-VL-7B)의 답을 확인해보세요.'
    )

    episode_images = gr.State([])

    with gr.Row():
      with gr.Column():
        random_btn = gr.Button('랜덤 에피소드 불러오기', variant='primary')
        video = gr.Image(label='에피소드 프레임 (GIF)')
      with gr.Column():
        ground_truth = gr.Markdown(label='데이터셋 정답 QA')
        question_box = gr.Textbox(label='자유 질문', placeholder='예: 로봇이 지금 무엇을 하고 있나요?')
        ask_btn = gr.Button('질문하기')
        answer_box = gr.Textbox(label='모델 답변', interactive=False)

    random_btn.click(
        load_random_episode,
        outputs=[video, ground_truth, episode_images, question_box, answer_box],
    )
    ask_btn.click(
        answer_question,
        inputs=[episode_images, question_box],
        outputs=[answer_box],
    )
    question_box.submit(
        answer_question,
        inputs=[episode_images, question_box],
        outputs=[answer_box],
    )

  demo.launch()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vlm-worker', action='store_true', help=argparse.SUPPRESS)
  args = parser.parse_args()
  if args.vlm_worker:
    run_vlm_worker()
  else:
    run_gradio_app()


if __name__ == '__main__':
  main()
