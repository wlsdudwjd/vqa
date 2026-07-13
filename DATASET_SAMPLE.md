# RoboVQA 데이터셋 샘플 (val split)

원본: [robovqa.github.io](https://robovqa.github.io) / [arXiv:2311.00899](https://arxiv.org/abs/2311.00899)

출처 파일: `json/val/data-00000-of-00021.json` (같은 데이터가 `tfrecord/val/`에도 바이너리로 존재)


VS Code에서 이 파일을 **마크다운 미리보기**(우측 상단 미리보기 아이콘, 또는 `Ctrl+Shift+V`)로 열면 아래 영상이 바로 재생됩니다.

---

## 샘플 1 — uid `13090847508390728210`

<video controls width="480" src="videos/2774825078398047015.mp4"></video>


| 태스크 타입 | 질문 | 정답 |
|---|---|---|
| `task:planning:freeform` | current goal is: Please remove the chips from the basket Q: immediate next step? | A: place the packet on the table |

---

## 샘플 2 — uid `2546174102638572251`

<video controls width="480" src="videos/12428962807005683550.mp4"></video>


| 태스크 타입 | 질문 | 정답 |
|---|---|---|
| `task:affordance:discriminative:discrete:True` | place the packet on the table Q: possible right now? | A: yes |
| `task:affordance:generative:positive:freeform` | Q: what action is possible right now? | A: place the packet on the table |

---

## 샘플 3 — uid `4343047547981987178`

<video controls width="480" src="videos/7394787237880323317.mp4"></video>


| 태스크 타입 | 질문 | 정답 |
|---|---|---|
| `task:affordance:discriminative:discrete:True` | place the packet on the table Q: possible right now? | A: yes |
| `task:immediate_planning_with_context20:freeform` | current goal is: Please remove the chips from the basket. last 20 steps: 1- place the packet on the table 2- place the packet on the table 3- place the packet on the table 4- place the packet on the table 5- place the packet on the table Q: immediate next step? | A: place the packet on the table |

---

## 샘플 4 — uid `7199094779446276165`

<video controls width="480" src="videos/7238758709431469220.mp4"></video>


| 태스크 타입 | 질문 | 정답 |
|---|---|---|
| `task:affordance:discriminative:discrete:False` | Place a brown sugar packet inside the organizer box Q: possible right now? | A: no |

---

## 샘플 5 — uid `14909867711120520341`

<video controls width="480" src="videos/9908030724488604675.mp4"></video>


| 태스크 타입 | 질문 | 정답 |
|---|---|---|
| `task:past_description:freeform` | Q: what just happened? | A: Place the orange inside the white bowl |

---

## 샘플 6 — uid `1854624089406269606`

<video controls width="480" src="videos/10984098182224562697.mp4"></video>


| 태스크 타입 | 질문 | 정답 |
|---|---|---|
| `task:future_prediction:freeform` | Q: what is likely to happen next? | A: Place the orange inside the white bowl |

---
