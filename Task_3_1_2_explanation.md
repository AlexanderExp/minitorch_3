# Разбор `tensor_map`

```python
size = int(np.prod(out_shape))
```
Сколько всего элементов надо обработать. Ровно столько будет итераций у внешней петли. np.prod - возврат произведения элементов по заданной оси. 

```python
aligned = True
...
for d in range(len(out_shape)):
    if out_shape[d] != in_shape[d] or out_strides[d] != in_strides[d]:
        aligned = False
        break
```
Проверка выровненности. Если одинаковы и shape, и strides — линейные адреса у `out` и `in` совпадают для одного и того же многомерного индекса.

### Если `aligned`

```python
for ord_ in prange(size):
    out_index = np.empty(MAX_DIMS, dtype=np.int32)
    rem = int(ord_)
    for i in range(len(out_shape) - 1, -1, -1):
        dim = int(out_shape[i])
        out_index[i] = rem % dim
        rem //= dim
    pos = index_to_position(out_index, out_strides)
    out[pos] = fn(float(in_storage[pos]))
```

* `prange(size)` — параллель по всем элементам.
* Создаём локальный буфер индексов `out_index` (размер фиксирован `MAX_DIMS`; реально используются первые `len(out_shape)` позиций).
* Раскладываем `ord_` в «смешанную систему счисления» по базам `out_shape[i]`. Это и есть многомерный индекс `out_index`.
* `pos = Σ out_index[i] * out_strides[i]` — линейный адрес.
* Поскольку `aligned=True`, адрес во входе тот же: читаем `in_storage[pos]`, применяем `fn` и пишем в `out[pos]`.


# 1) Как работает `prange(size)`?

`prange` — это *параллельный* вариант обычного `range`, который Numba понимает как «этот цикл можно распараллелить».

* Мы компилируем функцию с `njit(..., parallel=True)`.
* Компилятор Numba превращает цикл `for ord_ in prange(size):` в *parfor*: итерации делятся на куски и выполняются на нескольких потоках.
* Внутри такого цикла каждая итерация должна быть независимой: нет общих записей в один и тот же элемент, нет скрытых зависимостей.
* Локальные временные переменные и небольшие массивы (как наш `out_index`) **приватизируются** — у каждого потока/итерации свой экземпляр.
* Число потоков задаётся окружением (`NUMBA_NUM_THREADS`), планировщик выбирает размер чанка сам.

Итого: мы получаем простой способ «поэлементного» параллелизма по всем `size` элементам результата.

---

# 2) Как мы раскладываем `ord_` и получаем `pos`?

## 2.1. `ord_ → out_index` (смешанная система счисления)

`size = ∏ out_shape[d]`. Мы нумеруем все позиции результата от `0` до `size-1`.

Код:

```python
rem = int(ord_)
for i in range(len(out_shape) - 1, -1, -1):
    dim = int(out_shape[i])
    out_index[i] = rem % dim
    rem //= dim 
```

Пример: `out_shape = (2, 3, 4)`, `ord_ = 17`

* по оси 2 (размер 4): `17 % 4 = 1`, `rem = 17 // 4 = 4`
* по оси 1 (размер 3): `4 % 3 = 1`,  `rem = 4 // 3 = 1`
* по оси 0 (размер 2): `1 % 2 = 1`,  `rem = 1 // 2 = 0`
  → `out_index = (1, 1, 1)`

## 2.2. `out_index → pos` (линейный адрес)

Страйды (`out_strides`) хранят «шаг» в элементах по каждой оси. Линейный адрес — это скалярное произведение индекса на страйды:

```python
pos = Σ_{i=0..nd-1} out_index[i] * out_strides[i]
# у нас это делает index_to_position(out_index, out_strides)
```

Пример: если `out_shape=(2,3,4)`, то обычно `out_strides=(12,4,1)`
Для `out_index=(1,1,1)` получим: `pos = 1*12 + 1*4 + 1*1 = 17`.

---

# 3) Зачем нам `pos`?

Вся фактическая память тензора — это **одномерный** буфер `out`/`in_storage`.
Многомерные индексы существуют «логически», но чтобы прочитать/записать элемент, нужно знать *линейное смещение* в этом буфере — это и есть `pos`.

В aligned-ветке формы и страйды одинаковы у `out` и `in`, поэтому **адрес один и тот же**:

```python
out[pos] = fn( in_storage[pos] )
```

Это максимально дёшево:
* Одна раскладка индекса.
* Один расчёт адреса.
* Одно чтение, одно применение `fn`, одна запись.

Если бы мы не считали `pos`, нам пришлось бы делать сложные обращения по многомерным индексам (которых у массива на самом деле нет) или каждый раз заново сопоставлять `out_index` ↔ `in_index`. Страйды + линейный `pos` позволяют работать с произвольными представлениями памяти (contiguous/неcontiguous) **одинаковым кодом** и без накладных вызовов.

---

## Итог

* `prange(size)` распараллеливает «поэлементный» цикл.
* Маппинг `ord_ → out_index → pos` — это дешёвый способ узнать, **где** в плоском буфере лежит «наш» многомерный элемент.
* В aligned-случае `pos` совпадает у входа и выхода → минимум работы на итерацию и лучшая кэш-локальность.



### Если НЕ `aligned`

```python
for ord_ in prange(size):
    out_index = np.empty(MAX_DIMS, dtype=np.int32)
    in_index = np.empty(MAX_DIMS, dtype=np.int32)

    rem = int(ord_)
    for i in range(len(out_shape) - 1, -1, -1):
        dim = int(out_shape[i])
        out_index[i] = rem % dim
        rem //= dim

    broadcast_index(out_index, out_shape, in_shape, in_index)
    out_pos = index_to_position(out_index, out_strides)
    in_pos  = index_to_position(in_index,  in_strides)

    out[out_pos] = fn(float(in_storage[in_pos]))
```

* Снова раскладываем `ord_ → out_index`.
* `broadcast_index(...)` строит `in_index`:

  * если на оси `d` у входа размер `in_shape[d] == 1`, то `in_index[d] = 0`;
  * иначе `in_index[d] = out_index[d]`.
* Считаем два линейных адреса и делаем одну операцию.

## Почему мы не зовём `to_index` внутри цикла

`to_index` меняет свой аргумент `ordinal` (делит его на `dim` в цикле). Если инлайнить это прямо на переменной из `prange`, Numba воспринимает как **перезапись индекса цикла** и падает с `Overwrite of parallel loop index`. Поэтому мы делаем то же самое вручную, но на локальной копии `rem = int(ord_)`.

---












# Разбор `tensor_zip`

---

# Что делает `tensor_zip`

Это низкоуровневое ядро «побайтового»/«поэлементного» зипа двух тензоров:
для **каждого** выходного индекса `out_idx` оно вычисляет соответствующие индексы во входах `a_idx`, `b_idx` (с учётом broadcasting) и пишет:

```
out[out_pos] = fn(a[a_pos], b[b_pos])
```

где `*_pos` — линейные адреса по страйдам.

Возвращаемая функция работает напрямую с «сырыми» буферами и метаданными:

```
(out_storage, out_shape, out_strides,
 a_storage,   a_shape,   a_strides,
 b_storage,   b_shape,   b_strides) -> None
```

Функция компилируется через `njit(..., parallel=True)`, поэтому внешний цикл — параллельный.

---

## Механизм работы

### 1) Подсчёт количества итераций

```python
size = int(np.prod(out_shape))
```

Столько элементов нужно заполнить в `out`. Цикл пойдёт по `ord_ = 0..size-1`.

### 2) Проверка «выровненности» (fast-path)

```python
aligned = True
if not (len(out_shape) == len(a_shape) == len(b_shape)):
    aligned = False
else:
    for d in range(len(out_shape)):
        if (out_shape[d] != a_shape[d]
            or out_shape[d] != b_shape[d]
            or out_strides[d] != a_strides[d]
            or out_strides[d] != b_strides[d]):
            aligned = False
            break
```

**Смысл.** Если у `out`, `a`, `b` полностью совпадают формы и страйды, то один и тот же многомерный индекс даёт один и тот же линейный адрес `pos` во всех трёх буферах. Значит можно вообще не делать broadcasting и вторую адресацию: читать `a[pos]`, `b[pos]` и писать `out[pos]`.

### Случай fast-path

```python
for ord_ in prange(size):
    out_index = np.empty(MAX_DIMS, dtype=np.int32)
    rem = int(ord_)
    for i in range(len(out_shape) - 1, -1, -1):
        dim = int(out_shape[i])
        out_index[i] = rem % dim
        rem //= dim
    pos = index_to_position(out_index, out_strides)
    out[pos] = fn(float(a_storage[pos]), float(b_storage[pos]))
return
```

* `prange(size)` — параллелим по всем элементам.
* `ord_ → out_index` раскладываем в «смешанную систему счисления". Используем локальную копию `rem`, чтобы не «перезаписывать» индекс цикла (так безопаснее для parfors).
* `pos = Σ out_index[i] * out_strides[i]` — линейный адрес (функция `index_to_position` JIT-нута и инлайнится).
* Так как `aligned=True`, используем **один и тот же** `pos` для входов и выхода:

  ```python
  out[pos] = fn(a[pos], b[pos])
  ```

  Итог: один расчёт индекса, один адрес, два чтения, одно применение `fn`, одна запись.

## 3) Общая ветка (broadcasting/не совпадают страйды)

```python
for ord_ in prange(size):
    out_index = np.empty(MAX_DIMS, dtype=np.int32)
    a_index   = np.empty(MAX_DIMS, dtype=np.int32)
    b_index   = np.empty(MAX_DIMS, dtype=np.int32)

    # ord_ -> out_index
    rem = int(ord_)
    for i in range(len(out_shape) - 1, -1, -1):
        dim = int(out_shape[i])
        out_index[i] = rem % dim
        rem //= dim

    # сопоставляем индексы входов по правилам broadcasting
    broadcast_index(out_index, out_shape, a_shape, a_index)
    broadcast_index(out_index, out_shape, b_shape, b_index)

    # линейные адреса
    out_pos = index_to_position(out_index, out_strides)
    a_pos   = index_to_position(a_index,   a_strides)
    b_pos   = index_to_position(b_index,   b_strides)

    out[out_pos] = fn(float(a_storage[a_pos]), float(b_storage[b_pos]))
```

* Если формы/страйды не совпали, делаем broadcasting:

  * для каждой оси `d`:

    * если `a_shape[d] == 1` → `a_index[d] = 0` (ось «растянута»),
    * иначе `a_index[d] = out_index[d]`.
  * аналогично для `b`.
* Потом по своим страйдам каждого тензора считаем адреса: `a_pos`, `b_pos`, `out_pos`.
* Применяем `fn` к соответствующим элементам.

---

# Маленькие примеры

## 1) broadcast по первой оси: `a:(1,3)`, `b:(2,3)`, `out:(2,3)`

```
a_shape=(1,3)  a_strides=(3,1)  a_storage=[10,20,30]
b_shape=(2,3)  b_strides=(3,1)  b_storage=[ 1, 2, 3, 4, 5, 6]
out_shape=(2,3) out_strides=(3,1)
```

Итерации по `ord_`:

| ord\_ | out\_index | a\_index (broadcast) | b\_index | a\_pos | b\_pos | out\_pos | out = fn(a,b) |
| ----: | ---------- | -------------------- | -------- | -----: | -----: | -------: | ------------- |
|     0 | (0,0)      | (0,0)                | (0,0)    |      0 |      0 |        0 | fn(10,1)      |
|     1 | (0,1)      | (0,1)                | (0,1)    |      1 |      1 |        1 | fn(20,2)      |
|     3 | (1,0)      | (0,0)                | (1,0)    |      0 |      3 |        3 | fn(10,4)      |

…и т.д.

## 2) одинаковые формы, разные страйды (срез)

`a = base[:, ::2]` → `a_strides` по последней оси становится 2.

```
a_shape=(2,2), a_strides=(4,2)
b_shape=(2,2), b_strides=(2,1)
out_shape=(2,2), out_strides=(2,1)
```

Например, `ord_=1 → out_index=(0,1)`:

* `a_pos = 0*4 + 1*2 = 2`, `b_pos = 0*2 + 1*1 = 1`, `out_pos = 1`.
  Пишем `out[1] = fn(a[2], b[1])`.

---

# Сложность и память

* Время: `O(∏ out_shape)`; в общем случае на итерацию — константная работа адресации + 1 вызов `fn`.
* Память: только маленькие локальные буферы индексов (`MAX_DIMS`), которые Numba «вытаскивает» из цикла и приватизирует (нет лишних аллокаций на каждую итерацию).

---








# Разбор `tensor_zip`

## Назначение

`tensor_reduce(fn)` возвращает низкоуровневую функцию, которая сворачивает (reduce) входной тензор `a` по одной оси `reduce_dim` и пишет результат в `out`.
У `out_shape` всё совпадает с `a_shape`, кроме `reduce_dim`, где размер равен `1`.

Пример: если `a` имеет форму `(N, M, K)` и `reduce_dim=1`, то `out` будет `(N, 1, K)`.

---

# Сигнатура

```python
_reduce(
  out, out_shape, out_strides,   # куда пишем
  a_storage, a_shape, a_strides, # откуда читаем
  reduce_dim: int,               # какую ось сворачиваем
)
```

* `fn: (float, float) -> float` — операция (например, `add`, `mul`, `max`).
* Вызвавшая сторона уже заполнила `out` «стартовым значением» (например, 0 для суммы, 1 для произведения). Мы считаем это значение начальным `acc`.

---

# Предвычисления

```python
out_size   = int(np.prod(out_shape))        # сколько элементов нужно получить
red_len    = int(a_shape[reduce_dim])       # сколько идём по оси свёртки
red_stride = int(a_strides[reduce_dim])     # шаг в памяти на +1 по reduce_dim
```

* `red_stride` позволяет не считать позицию заново в каждом шаге внутреннего цикла — мы просто увеличиваем адрес на фиксированный шаг.

---

# Внешний цикл = parfor

```python
for ord_ in prange(out_size):
```

* Параллелим по всем выходным элементам — каждая итерация пишет в уникальный `out_pos`.

---

# Декодирование логического индекса

```python
out_index = np.empty(MAX_DIMS, np.int32)
a_index   = np.empty(MAX_DIMS, np.int32)

# ord_ -> out_index  (как unravel_index)
rem = int(ord_)
for i in range(len(out_shape) - 1, -1, -1):
    dim = int(out_shape[i])
    out_index[i] = rem % dim
    rem //= dim
```

* Это превращает одномерный `ord_` в многомерные координаты `out_index` по `out_shape`.

---

# Адрес в `out`

```python
out_pos = index_to_position(out_index, out_strides)
```

* Линейный адрес = `Σ out_index[d] * out_strides[d]`.

---

# «База» в `a` для этой позиции

```python
# a_index := out_index, но reduce_dim = 0
for d in range(len(a_index)):
    a_index[d] = out_index[d]
a_index[reduce_dim] = 0

a_base = index_to_position(a_index, a_strides)
```

* Идея: у текущего «среза» (фиксируем все оси ≠ `reduce_dim` по `out_index`) мы хотим пройтись вдоль оси `reduce_dim` от `0` до `red_len-1`.
* `a_base` — адрес элемента с `reduce_dim=0`. Дальше мы будем просто идти по памяти шагом `red_stride`.

---

# Внутренний цикл свёртки (последовательный и локальный)

```python
acc = float(out[out_pos])  # стартовое значение (0 для sum, 1 для mul, ...)
pos = a_base
for _ in range(red_len):
    acc = fn(acc, float(a_storage[pos]))
    pos += red_stride         # двигаемся по оси свёртки
out[out_pos] = acc            # единственная глобальная запись
```

* Внутри цикла нет глобальных записей (только локальная переменная `acc`) → это безопасно для параллелизма.
* Мы не вызываем индексные функции в этом внутреннем цикле — только линейный инкремент адреса `pos += red_stride`. Это дёшево.

# Небольшие примеры

## 1) Простая сумма по последней оси

`a` (C-порядок):

```
a_shape=(2,3), a_strides=(3,1)
a_storage = [1,2,3,  4,5,6]  # строки: [1,2,3] и [4,5,6]
reduce_dim = 1
start = 0
out_shape=(2,1), out_strides=(1,1) # допустим
```

* `out_size = 2*1 = 2`, `red_len = 3`, `red_stride = 1`.
* Итерация `ord_=0`:

  * `out_index=(0,0)` → `out_pos` (допустим) = 0.
  * `a_index=(0,0)` → `a_base=0`.
  * `acc = out[0] = 0`. Идём `red_len=3` шагами `+1`:

    * pos=0 → `acc=0+1=1`
    * pos=1 → `acc=1+2=3`
    * pos=2 → `acc=3+3=6`
  * `out[0]=6`.
* Итерация `ord_=1`:

  * `out_index=(1,0)` → `a_index=(1,0)` → `a_base=3`.
  * `acc=0`:

    * pos=3 → `+4=4`
    * pos=4 → `+5=9`
    * pos=5 → `+6=15`
  * `out[1]=15`.

Итог: `out = [[6],[15]]`.

## 2) Свёртка по последней оси

```
base_shape=(2,4), base_strides=(4,1)
base = [0,1,2,3, 4,5,6,7]

a = base[:, ::2]  # берём каждые 2 колонки
a_shape=(2,2), a_strides=(4,2)  # шаг по последней оси = 2
reduce_dim=1, start=0
out_shape=(2,1)
```

* Для строки 0: `a_base = 0`, `red_len=2`, `red_stride=2` → читаем `base[0]` и `base[2]`: `0 + 2 = 2`.
* Для строки 1: `a_base = 4` → читаем `base[4]` и `base[6]`: `4 + 6 = 10`.
* Итог `out = [[2],[10]]`.
  Видно, что формула «адрес = базовый + k \* stride» прекрасно работает и при срезах.

---

# Итого

* `prange(out_size)` — параллель по всем выходным элементам.
* `ord_ → out_index` — логический индекс текущего элемента.
* `out_pos` — адрес записи (по страйдам `out`).
* `a_index := out_index; a_index[reduce_dim]=0` — базовая точка «среза» в `a`.
* `a_base` — начальный адрес чтения.
* Внутренний цикл: идём `red_len` раз шагом `red_stride`, накапливаем `acc` через `fn`.
* В конце одной итерации — **ровно одна глобальная запись** `out[out_pos] = acc`.
