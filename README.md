# Battery Anomaly Detection (DyAD + PrefixSpan)

Проект для обнаружения аномалий в данных аккумуляторных батарей электромобилей.
Включает три независимых, но связанных модуля:

1. **Подготовка данных** — сбор информации об автомобилях и разбиение на 5 фолдов.
2. **Модель DyAD** — обучение автоэнкодера (GRU/LSTM) для восстановления сигналов
   и оценки аномальности на уровне автомобиля (vehicle-level AUROC).
3. **Интерпретация** — извлечение интерпретируемых паттернов аномального поведения
   через дискретизацию признаков и алгоритм PrefixSpan.

---

## 📁 Структура проекта

```
.
├── my_loading_data.py              # Подготовка данных и фолдов
├── my_dyad.py                      # Модель DyAD и оценка AUROC
├── my_prefix_span_interpretation.py# Интерпретация через PrefixSpan
└── README.md
```

---

## 🔧 Требования

- Python 3.9+
- PyTorch
- NumPy, pandas, scikit-learn
- tqdm, natsort
- `prefixspan`
- `matplotlib`

Установка:

```bash
pip install torch numpy pandas scikit-learn tqdm natsort prefixspan matplotlib
```

---

## 📊 Формат данных

Ожидается, что данные лежат в каталоге `../data/battery_brand*/` и представляют собой
`.pkl`-файлы. Каждый файл содержит кортеж `(data_list, metadata)`, где:

- `data_list` — список строк с колонками
  `['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp', 'min_temp', 'timestamp']`
- `metadata` — словарь с ключами `'car'`, `'charge_segment'`, `'label'`, `'mileage'`
  (метка `'00'` — норма, иначе — аномалия).

---

## 🚀 Использование

### 1. Подготовка данных (`my_loading_data.py`)

Собирает все `.pkl`-файлы по брендам, группирует их по автомобилям и сохраняет:

- `five_fold_utils/ind_odd_dict{1,2,3}.npz` — списки нормальных (`ind`) и
  аномальных (`ood`) автомобилей для каждого бренда;
- `five_fold_utils/all_car_dict.npz` — словарь `car_id -> [пути к .pkl]`.

```bash
python my_loading_data.py
```

### 2. Обучение и оценка DyAD (`my_dyad.py`)

```bash
python my_dyad.py \
    --all_car_dict ../five_fold_utils/all_car_dict.npz.npy \
    --ind_ood_dict ../five_fold_utils/ind_odd_dict3.npz.npy \
    --fold 0 \
    --epochs 30 \
    --batch_size 128 \
    --enc_idx 0 1 2 3 4 5 \
    --tgt_idx 2 3 4 5
```

Ключевые аргументы:

| Аргумент | Описание |
|---|---|
| `--all_car_dict` | путь к словарю автомобилей |
| `--ind_ood_dict` | путь к словарю in-distribution / OOD |
| `--fold` | номер фолда (0–4) |
| `--epochs` | число эпох обучения |
| `--batch_size` | размер батча |
| `--enc_idx` | индексы признаков, подаваемых в энкодер |
| `--tgt_idx` | индексы признаков, которые восстанавливаются |

Скрипт выводит `AUROC (vehicle-level)` — качество разделения нормы и аномалий
по максимальной ошибке восстановления на автомобиль (top-10 % сниппетов).

### 3. Интерпретация паттернов (`my_prefix_span_interpretation.py`)

Скрипт:

1. Загружает сниппеты и группирует их по автомобилям и сегментам заряда.
2. Обучает квантильные дискретизаторы (LOW / NORMAL / HIGH) на **нормальных** данных.
3. Превращает каждый сниппет в символьное событие (например, `V_HIGH,SOC_LOW`).
4. Строит символьные последовательности на автомобиль.
5. Находит частые паттерны в аномальных последовательностях, которых нет в нормальных
   (PrefixSpan, `closed=True`).
6. Сохраняет отчёт в `../data/<brand>/patterns_report.xlsx`.

```bash
python my_prefix_span_interpretation.py
```

> В блоке `if __name__ == "__main__"` задаются `brand` и `FOLDER_PATH` — поменяйте
> их под свой датасет.

---

## 🧠 Модель DyAD (кратко)

Архитектура:

- **Encoder** (GRU/LSTM) → `mu`, `logvar` (латентное представление).
- **Decoder** (GRU/LSTM) инициализируется из `z` и восстанавливает целевую часть сигнала.
- **Mileage head** — вспомогательная регрессия пробега (регуляризация латента).

Функция потерь:

```
L = MSE(x_hat, x) + w_kl * KL(q(z|x) || N(0, I)) + w_mileage * MSE(mileage_pred, mileage)
```

Оценка аномальности: средняя ошибка восстановления по `top_p %` худших сниппетов
на автомобиль.

---

## 📈 Результаты

После запуска всех трёх скриптов вы получите:

- разбиение на фолды (`.npz`),
- обученную модель DyAD и метрику AUROC,
- Excel-отчёт с интерпретируемыми паттернами аномалий.
