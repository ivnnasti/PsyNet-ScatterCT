# PhysScatterNet

Коррекция артефактов рассеяния в КБКТ с помощью физически-обоснованной нейросети на базе Residual U-Net.

Сеть обучается переводить КБКТ-сканы (с артефактами рассеяния) в парные КТ-сканы (без артефактов), используя комбинированную функцию потерь: L1 + SSIM + физический штраф на неотрицательность рассеяния.

---

## Структура проекта

```
D:\Project\PhysNet-ScatterCT\
├── data\
│   ├── TRAINCBCTSimulated\
│   │   ├── 32\      ← КБКТ с 32 проекциями  (uint16 .nii)
│   │   ├── 64\
│   │   ├── 128\     ← используется по умолчанию
│   │   ├── 256\
│   │   └── 490\     ← наибольшее число проекций, меньше артефактов
│   ├── TRAINCTAlignedToCBCT\   ← парные КТ ground truth (float32 .nii, HU)
│   └── TRAINMasksAlignedToCBCT\
│       ├── masks\   ← 3 класса: 0=фон, 1=мягкие ткани, 2=кость
│       └── masks01\ ← бинарные маски
├── src\
│   ├── dataset.py       — загрузка и нормализация данных
│   ├── model.py         — Residual U-Net
│   ├── loss.py          — комбинированная функция потерь
│   ├── train.py         — цикл обучения
│   ├── evaluate.py      — оценка метрик на val-наборе
│   └── plot_training.py — график кривых обучения
├── templates\
│   └── index.html       — веб-интерфейс
├── static\
│   ├── css\style.css
│   └── js\app.js
├── output\              — генерируется при обучении
│   ├── best_model.pth
│   ├── train_log.json
│   ├── metrics.json
│   └── training_curves.png
├── app.py               — Flask-сервер
└── requirements.txt
```

---

## Данные

| Параметр | Значение |
|---|---|
| Число пациентов | 131 |
| Форма тома | (366, 238, 364) вокселей |
| Вокселный размер | 0.69 × 1.03 × 0.69 мм |
| КБКТ формат | uint16, [0, 65535] |
| КТ формат | float32, HU [-1024, 1500] |
| Разбивка train/val | 80/20 по пациентам (не по срезам) |

Нормализация перед подачей в сеть:
- КБКТ: `x / 32767.5 − 1` → [-1, 1]
- КТ: `clip(x, -1024, 1500)` → `(x − 238) / 1262` → [-1, 1]

Разрешение КБКТ задаётся через `CBCT_RES` (папка 32/64/128/256/490). По умолчанию `128`.

---

## Установка

### Требования

- Python 3.9+
- CPU достаточен (модель лёгкая, базовые каналы 4 по умолчанию)
- GPU опционален (поддержка через PyTorch)

### Установка зависимостей

```bash
cd D:\Project\PhysNet-ScatterCT
pip install -r requirements.txt
```

`requirements.txt`:
```
torch
nibabel
numpy
matplotlib
scikit-image
flask
```

---

## Запуск обучения

```bash
cd D:\Project\PhysNet-ScatterCT
python src/train.py
```

Пример вывода:
```
ep 1/20 | train 0.4321 | val 0.4105
ep 2/20 | train 0.3812 | val 0.3640
...
```

Лучшая модель сохраняется в `output/best_model.pth` при обновлении минимального val loss.
История потерь сохраняется в `output/train_log.json`.

### Параметры обучения (переменные окружения)

Есть готовые профили запуска через `PROFILE`:

| PROFILE | Для чего | Значения |
|---|---|---|
| `fast` | быстрый запуск на ноутбуке | `ch=4, step=16, size=96, bs=8, epochs=5` |
| `balanced` | режим по умолчанию | `ch=4, step=8, size=96, bs=8, epochs=8` |
| `full` | медленнее, но качественнее | `ch=8, step=4, size=128, bs=4, epochs=20` |

Если `PROFILE` задан, эти значения подставляются автоматически. Любую отдельную переменную всё равно можно переопределить вручную.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PROFILE` | `balanced` | Готовый профиль: `fast`, `balanced`, `full` |
| `EPOCHS` | `8` | Число эпох |
| `BS` | `8` | Размер батча обучения |
| `LR` | `1e-3` | Скорость обучения (Adam) |
| `CH` | `4` | Базовые каналы модели |
| `STEP` | `8` | Брать каждый N-й аксиальный срез |
| `SIZE` | `96` | Ресайз среза до SIZE×SIZE пикселей |
| `MAX_VOLS` | `0` | Лимит томов (0 = все 131) |
| `VAL_BS` | `16` | Размер батча валидации |
| `CBCT_RES` | `128` | Разрешение КБКТ (32/64/128/256/490) |
| `DATA_DIR` | автоопределение | Путь к папке данных |

При настройках по умолчанию (STEP=8, SIZE=96, все 131 том):
- ~4 700 обучающих срезов, ~1 200 val-срезов
- заметно меньше RAM и быстрее старт
- обычно быстрее на обычном CPU ноутбука

**Быстрый тест (2–5 мин всего):**
```powershell
$env:PROFILE='fast'; $env:MAX_VOLS=10; python src/train.py
```

**Ноутбук, обычный запуск:**
```powershell
$env:PROFILE='fast'; python src/train.py
```

**Сбалансированный режим:**
```powershell
$env:PROFILE='balanced'; python src/train.py
```

Во время обучения теперь показываются строки вида:
```text
ep 1/8 start
	ep 1/8  batch 50/592  loss 0.2143  eta 42s
	ep 1/8  batch 100/592  loss 0.2011  eta 37s
ep 1/8 | train 0.1987 | val 0.1812 | 74s
```

**Полное обучение (Windows PowerShell):**
```powershell
$env:PROFILE='full'; $env:EPOCHS=50; $env:CBCT_RES=256; python src/train.py
```

**Linux / macOS:**
```bash
PROFILE=full EPOCHS=50 CBCT_RES=256 python src/train.py
```

---

## График кривых обучения

После завершения обучения:

```bash
python src/plot_training.py
```

Сохраняет `output/training_curves.png`.

---

## Оценка метрик

```bash
python src/evaluate.py
```

Пример вывода:
```
rmse: 0.0842 | ssim: 0.8731
```

Сохраняет `output/metrics.json`:
```json
{"rmse": 0.0842, "ssim": 0.8731}
```

---

## Веб-интерфейс

```bash
python app.py
```

Открыть в браузере: [http://127.0.0.1:5000](http://127.0.0.1:5000)

Возможности интерфейса:
- Просмотр метрик RMSE и SSIM
- График кривых обучения
- Загрузка файла `.nii` / `.nii.gz` → инференс → отображение входного среза и результата коррекции
- Переключение тёмной/светлой темы

### API маршруты

| Метод | URL | Описание |
|---|---|---|
| GET | `/` | Веб-интерфейс |
| GET | `/metrics` | JSON с RMSE и SSIM |
| GET | `/plot` | График `training_curves.png` |
| POST | `/predict` | Загрузить `.nii.gz`, получить base64 PNG |

---

## Архитектура модели

Residual U-Net с базовыми каналами `ch=4` по умолчанию:

```
Encoder:  [1 → ch] → [ch → 2ch] → [2ch → 4ch]  (MaxPool2d после каждого блока)
Bottleneck: [4ch → 8ch]
Decoder:  [8ch+4ch → 4ch] → [4ch+2ch → 2ch] → [2ch+ch → ch]  (bilinear upsample + skip)
Output:   Conv2d(ch, 1, 1) + Tanh
```

Каждый блок: `Conv → BN → ReLU → Conv → BN + residual skip → ReLU`

---

## Функция потерь

```
L = L1(pred, CT) + 0.5 × (1 − SSIM(pred, CT)) + 0.1 × ReLU(pred − CT).mean()
```

Физический штраф: рассеяние `scatter = CT − pred` должно быть ≥ 0 (CT не может быть меньше скорректированного КБКТ).

---

## Рекомендуемый порядок запуска

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Обучить модель
python src/train.py

# 3. Построить график кривых обучения
python src/plot_training.py

# 4. Посчитать метрики
python src/evaluate.py

# 5. Запустить веб-интерфейс
python app.py
```
