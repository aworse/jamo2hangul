# jamo2hangul — 자모 → 한글 복원 모델 (v2.0)

> **English summary.** jamo2hangul restores spaced Korean text from a flat stream of jamo (Korean letters), e.g. `ㅎㅏㄴㄱㅡㄹ` → `한글`. It is a character-level encoder–decoder Transformer (~53M parameters; d_model 512, 8 heads, 6+6 layers) trained from scratch on about 2.06M Korean Wikipedia sentences, built for a Korean speech-recognition project. On 8,000 held-out words, the clean model gets 96.70% exactly right (7,736/8,000) with a 1.36% character error rate. A noise-fine-tuned model cuts the character error rate on corrupted input from 32.8% to 14.9%. Model weights are not published yet and are available on request (bagdoeon827@gmail.com).
>
> Author: Bugeon Park ([@Bugeon151](https://github.com/Bugeon151); aworse is my second GitHub account).

평평한 호환 자모 스트림(`ㅎㅏㄴㄱㅡㄹ`)을 받아 **띄어쓰기가 포함된 한글**(`한글`)로 복원하는
char 단위 Transformer seq2seq. 한국어 문장을 자모로 분해해 학습 데이터를 무한 생성하므로 별도
데이터 없이 학습된다. 입력 자모에 오류가 섞여도 교정하도록 노이즈 주입 파인튜닝 버전을 함께 제공한다.

- **인코더-디코더 Transformer** (d_model 512 / 8 heads / 6+6 layers, ~53M params)
- 학습 코퍼스: 한국어 위키백과 약 **2,058,322 문장**
- 두 가지 모델: **clean**(정상 입력 최적) / **noise**(오류 섞인 입력에 강함)

## 성능 (v2.0)

### 단어 정확도 — 8,000개 held-out 단어 (clean 모델)
| 지표 | 값 |
|---|---|
| 완전일치(word accuracy) | **96.70%** (7,736/8,000) |
| 문자오류율(CER) | **1.36%** |

길이별: 1–4음절 **97–99%**, 5음절 93%, 6음절+ 하락(대부분 복합어 띄어쓰기 분리).
틀린 사례의 다수는 글자 오류가 아니라 복합어를 띄어 쓴 경우(쇼핑센터→쇼핑 센터)이며,
띄어쓰기를 무시한 순수 음절 정확도는 약 99%+로 추정된다. 나머지는 희귀 한자(漢字).

### 커스텀 예시 (단어·주소·문장·숫자/영문 혼합, 25개)
| 입력 | clean 모델 | noise 모델 |
|---|---|---|
| 깨끗한 입력 — exact | 88.0% | 88.0% |
| 깨끗한 입력 — CER | **1.70%** | 2.13% |
| 오류 섞인 입력 — exact | 32.0% | **48.0%** |
| 오류 섞인 입력 — CER | 32.8% | **14.9%** |

- 단어·주소·고유명사(깨끗한 입력)는 **100% 일치** (부산광역시 해운대구 좌동 등)
- 노이즈 파인튜닝으로 **오류 입력 CER이 절반 이하**로 개선, 깨끗한 입력 성능은 거의 유지

### 개발셋 (위키 held-out 5,000문장, clean best)
CER **2.05%**, exact 47.4% (문장 전체·띄어쓰기 포함 완전일치 기준)

## 모델 파일

가중치 파일은 아직 공개 릴리스로 올리지 않았습니다. 필요하면 bagdoeon827@gmail.com으로 요청해 주세요.

| 파일 | 설명 |
|---|---|
| `jamo2hangul_clean_v2.pt` | clean 모델 (정상 입력 최적), fp16 inference용 |
| `jamo2hangul_noise_v2.pt` | noise 모델 (오류 입력에 강함), fp16 inference용 |

## 추론

```bash
# vocab.json 이 있는 디렉터리(dir)와 체크포인트를 지정
python train.py infer --dir artifacts/run --ckpt jamo2hangul_clean_v2.pt --in jamo.txt --out hangul.txt
```

입력(`jamo.txt`)은 한 줄에 한 샘플, 자모 또는 한글 음절 모두 가능. 공백은 자동 제거 후 모델이 복원.

## 재현 / 재학습

```bat
run.bat                      ::  설치 → 위키 다운로드·전처리 → 학습 → 평가 (Windows, NVIDIA GPU)
run.bat --noise 0.05,0.03,0.03   ::  오류 교정(노이즈) 학습
```

## 구성
| 파일 | 내용 |
|---|---|
| `jamo_text.py` | 한글↔호환자모 분해/합성, 쌍 생성, 노이즈 주입 |
| `data.py` | Vocab, 토치 Dataset, 토큰 예산 배칭 |
| `model.py` | char Transformer 인코더-디코더, greedy 디코딩 |
| `train.py` | `prepare` / `train` / `eval` / `infer` CLI (bf16, 체크포인트/재개, CER·exact) |
| `eval_examples.py`, `word_eval.py` | 커스텀 예시 / 대규모 단어 평가 |
| `run_j2h.py`, `run.bat` | 설치·다운로드·학습 실행기 |

## 평가 지표
- **exact**: 정답 문자열 완전 일치율(띄어쓰기 포함)
- **CER**: 문자 오류율(편집거리 / 정답 길이, 띄어쓰기 포함)

## 메모
- 기본 입력은 자모 사이 공백·단어 경계가 없다고 보고 띄어쓰기를 모델이 복원한다.
- 대회 실제 데이터로는 아직 파인튜닝하지 않은 **범용 베이스라인**이다. 대회 입력 형식·오류 분포에
  맞춰 파인튜닝하면 성능이 더 오를 여지가 크다.
