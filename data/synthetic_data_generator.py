# -*- coding: utf-8 -*-
"""
Генератор синтетических данных для проекта
«Аналитика IT-поддержки аэропорта Домодедово».

Данные полностью вымышленные, но смоделированы так, чтобы повторять
поведение реальной службы технической поддержки крупного аэропорта:

  * суточные пики обращений совпадают с волнами вылетов и прилётов;
  * недельная и годовая сезонность (летний пик, новогодние праздники);
  * рост нагрузки год к году и старение части оборудования;
  * нарушения SLA учащаются в дни пиковой нагрузки;
  * критические инциденты (P1) на ключевых системах приводят к
    простоям и задержкам рейсов.

Запуск:
    python3 data/synthetic_data_generator.py

Результат (папки создаются автоматически):
    data/raw/tickets.csv           — журнал обращений (факт-таблица)
    data/raw/flights_daily.csv     — суточные объёмы рейсов и пассажиров
    data/reference/sla_policy.csv  — целевые показатели SLA по приоритетам
    data/reference/systems.csv     — справочник систем аэропорта
    data/reference/agents.csv      — справочник сотрудников поддержки
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Параметры воспроизводимости и периода
# ----------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")
BASE_DIR = Path(__file__).resolve().parent            # .../data
RAW_DIR = BASE_DIR / "raw"
REF_DIR = BASE_DIR / "reference"
RAW_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Справочники: системы аэропорта
# ----------------------------------------------------------------------------
# (название, критичность, базовый вес частоты обращений, влияет ли на рейсы)
SYSTEMS = [
    ("Табло вылета и прилёта (FIDS)",      "Высокая",     0.09, True),
    ("Стойки и киоски регистрации",         "Высокая",     0.12, True),
    ("Система обработки багажа (BHS)",      "Критическая", 0.05, True),
    ("Посадочные гейты",                    "Высокая",     0.08, True),
    ("Пассажирский Wi-Fi и сеть",           "Средняя",     0.10, False),
    ("Кассовые системы (POS)",              "Средняя",     0.07, False),
    ("Видеонаблюдение (CCTV)",              "Средняя",     0.06, False),
    ("Громкая связь (PA)",                  "Средняя",     0.04, False),
    ("Рабочие станции (АРМ)",               "Низкая",      0.16, False),
    ("Принтеры посадочных талонов",         "Средняя",     0.08, True),
    ("Система контроля доступа (СКУД)",     "Высокая",     0.05, True),
    ("Серверы и ЦОД",                       "Критическая", 0.04, True),
    ("Телефония",                           "Низкая",      0.06, False),
]
SYS_NAMES = [s[0] for s in SYSTEMS]
SYS_CRIT = {s[0]: s[1] for s in SYSTEMS}
SYS_WEIGHT = np.array([s[2] for s in SYSTEMS], dtype=float)
SYS_WEIGHT = SYS_WEIGHT / SYS_WEIGHT.sum()
SYS_CRITICAL_FOR_FLIGHTS = {s[0]: s[3] for s in SYSTEMS}

# Категория обращения зависит от системы (распределение по каждой системе)
CATEGORIES = ["Оборудование", "ПО", "Сеть", "Периферия", "Доступы"]
CAT_BY_SYS = {
    "Табло вылета и прилёта (FIDS)":   [0.30, 0.40, 0.30, 0.00, 0.00],
    "Стойки и киоски регистрации":      [0.40, 0.30, 0.10, 0.15, 0.05],
    "Система обработки багажа (BHS)":   [0.50, 0.30, 0.20, 0.00, 0.00],
    "Посадочные гейты":                 [0.30, 0.40, 0.30, 0.00, 0.00],
    "Пассажирский Wi-Fi и сеть":        [0.15, 0.05, 0.78, 0.02, 0.00],
    "Кассовые системы (POS)":           [0.15, 0.40, 0.20, 0.25, 0.00],
    "Видеонаблюдение (CCTV)":           [0.50, 0.10, 0.40, 0.00, 0.00],
    "Громкая связь (PA)":               [0.60, 0.10, 0.30, 0.00, 0.00],
    "Рабочие станции (АРМ)":            [0.35, 0.30, 0.05, 0.10, 0.20],
    "Принтеры посадочных талонов":      [0.20, 0.05, 0.10, 0.65, 0.00],
    "Система контроля доступа (СКУД)":  [0.30, 0.10, 0.15, 0.00, 0.45],
    "Серверы и ЦОД":                    [0.25, 0.40, 0.35, 0.00, 0.00],
    "Телефония":                        [0.50, 0.10, 0.40, 0.00, 0.00],
}

# Основная зона обращения по системе
PRIMARY_ZONE = {
    "Табло вылета и прилёта (FIDS)":   "Зона вылета и гейты",
    "Стойки и киоски регистрации":      "Зона регистрации",
    "Система обработки багажа (BHS)":   "Выдача багажа",
    "Посадочные гейты":                 "Зона вылета и гейты",
    "Пассажирский Wi-Fi и сеть":        "Зона вылета и гейты",
    "Кассовые системы (POS)":           "Торговые зоны",
    "Видеонаблюдение (CCTV)":           "Служебные зоны",
    "Громкая связь (PA)":               "Зона вылета и гейты",
    "Рабочие станции (АРМ)":            "Служебные зоны",
    "Принтеры посадочных талонов":      "Зона регистрации",
    "Система контроля доступа (СКУД)":  "Предполётный досмотр",
    "Серверы и ЦОД":                    "ЦОД и серверная",
    "Телефония":                        "Служебные зоны",
}
ALL_ZONES = sorted(set(PRIMARY_ZONE.values()) | {
    "Прилёт и паспортный контроль", "Перрон (RAMP)"
})
ZONE_ROLE = {
    "Зона регистрации": "Авиакомпания",
    "Предполётный досмотр": "Служба безопасности",
    "Зона вылета и гейты": "Наземное обслуживание",
    "Прилёт и паспортный контроль": "Служба безопасности",
    "Выдача багажа": "Наземное обслуживание",
    "Перрон (RAMP)": "Наземное обслуживание",
    "Торговые зоны": "Торговая точка",
    "Служебные зоны": "Администрация аэропорта",
    "ЦОД и серверная": "Операционный центр",
}
ALL_ROLES = sorted(set(ZONE_ROLE.values()) | {"Пассажирский сервис"})

# ----------------------------------------------------------------------------
# Справочник SLA по приоритетам (документированная политика)
# ----------------------------------------------------------------------------
PRIORITIES = ["P1", "P2", "P3", "P4"]
PRIORITY_NAME = {"P1": "Критический", "P2": "Высокий", "P3": "Средний", "P4": "Низкий"}
RESP_TARGET = {"P1": 15, "P2": 30, "P3": 60, "P4": 240}          # минуты
RES_TARGET = {"P1": 240, "P2": 300, "P3": 480, "P4": 1080}       # минуты

# Распределение приоритетов по критичности системы
PRIORITY_BY_CRIT = {
    "Критическая": [0.10, 0.30, 0.45, 0.15],
    "Высокая":     [0.03, 0.20, 0.55, 0.22],
    "Средняя":     [0.01, 0.12, 0.57, 0.30],
    "Низкая":      [0.004, 0.06, 0.556, 0.38],
}

# Каналы обращения по приоритету
CHANNELS = ["Телефон", "Электронная почта", "Портал самообслуживания",
            "Личное обращение", "Мониторинг"]
CHANNEL_BY_PRIO = {
    "P1": [0.50, 0.05, 0.00, 0.10, 0.35],
    "P2": [0.45, 0.15, 0.10, 0.15, 0.15],
    "P3": [0.35, 0.20, 0.25, 0.15, 0.05],
    "P4": [0.15, 0.30, 0.40, 0.15, 0.00],
}

# ----------------------------------------------------------------------------
# Справочник команд и сотрудников поддержки
# ----------------------------------------------------------------------------
TEAMS = {
    "1-я линия поддержки": 11,
    "2-я линия / Сети": 5,
    "2-я линия / Серверы и ПО": 5,
    "Полевая служба": 7,
    "Вендор/подрядчик": 4,
}
agents_rows = []
team_to_agents = {}
_counter = 100
for team, n in TEAMS.items():
    ids = []
    for _ in range(n):
        _counter += 1
        aid = f"AG-{_counter}"
        ids.append(aid)
        agents_rows.append({"agent_id": aid, "team": team})
    team_to_agents[team] = ids
agents_df = pd.DataFrame(agents_rows)

# Назначение команды по категории (с учётом эскалации по приоритету)
TEAM_BY_CAT = {
    "Сеть":        (["2-я линия / Сети", "1-я линия поддержки"], [0.70, 0.30]),
    "Оборудование": (["Полевая служба", "1-я линия поддержки", "Вендор/подрядчик"], [0.55, 0.35, 0.10]),
    "ПО":          (["1-я линия поддержки", "2-я линия / Серверы и ПО", "Вендор/подрядчик"], [0.50, 0.35, 0.15]),
    "Периферия":   (["1-я линия поддержки", "Полевая служба"], [0.60, 0.40]),
    "Доступы":     (["1-я линия поддержки", "2-я линия / Серверы и ПО"], [0.80, 0.20]),
}

# ----------------------------------------------------------------------------
# Шаг 1. Суточные объёмы рейсов и пассажиров
# ----------------------------------------------------------------------------
dates = pd.date_range(DATE_START, DATE_END, freq="D")
n_days = len(dates)

# Годовая сезонность по месяцам (индекс 1..12)
MONTH_FACTOR = {1: 1.05, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,
                7: 1.20, 8: 1.22, 9: 1.05, 10: 0.98, 11: 0.92, 12: 1.12}
# Недельная сезонность (Пн=0..Вс=6)
WEEKDAY_FACTOR = np.array([1.06, 1.03, 1.02, 1.03, 1.08, 0.96, 0.92])

month_f = np.array([MONTH_FACTOR[d.month] for d in dates])
weekday_f = WEEKDAY_FACTOR[dates.dayofweek.values]
# Плавный рост объёмов год к году (+9% за 2 года)
trend_f = 1.0 + 0.09 * (np.arange(n_days) / n_days)

BASE_FLIGHTS = 600
flights_noise = rng.normal(1.0, 0.05, n_days)
flights = BASE_FLIGHTS * month_f * weekday_f * trend_f * flights_noise
flights = np.round(flights).astype(int)

avg_pax_per_flight = rng.normal(138, 6, n_days)
load_factor_pax = rng.normal(0.82, 0.04, n_days).clip(0.6, 0.98)
passengers = np.round(flights * avg_pax_per_flight * load_factor_pax).astype(int)

flights_daily = pd.DataFrame({
    "date": dates.date,
    "flights_scheduled": flights,
    "passengers": passengers,
    "day_of_week": dates.dayofweek.map(
        {0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"}),
    "is_weekend": dates.dayofweek.isin([5, 6]),
    "month": dates.month,
    "year": dates.year,
})

# ----------------------------------------------------------------------------
# Шаг 2. Суточное число обращений (зависит от объёма рейсов)
# ----------------------------------------------------------------------------
flights_mean = flights.mean()
# База обращений связана с загрузкой аэропорта + собственный IT-шум
it_month_noise = rng.normal(1.0, 0.04, n_days)
daily_lambda = 40 * (0.30 + 0.85 * flights / flights_mean) * it_month_noise
daily_counts = rng.poisson(daily_lambda)

# Коэффициент нагрузки дня: влияет на время реакции/решения и на нарушения SLA
load_ratio = daily_counts / np.median(daily_counts)
load_factor_day = np.clip(load_ratio ** 0.6, 0.8, 1.7)
load_by_date = dict(zip(dates.date, load_factor_day))

# ----------------------------------------------------------------------------
# Шаг 3. Разворачиваем обращения по дням и часам
# ----------------------------------------------------------------------------
# Суточный профиль по часам (две волны: утро и вечер)
HOUR_WEIGHTS = np.array([
    0.20, 0.15, 0.12, 0.15, 0.35, 0.80, 1.60, 2.00, 1.90, 1.50, 1.20, 1.10,
    1.05, 1.00, 1.00, 1.10, 1.40, 1.80, 1.90, 1.70, 1.30, 0.90, 0.60, 0.35])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()

created_at_list = []
for day, cnt in zip(dates, daily_counts):
    if cnt == 0:
        continue
    hrs = rng.choice(24, size=cnt, p=HOUR_WEIGHTS)
    mins = rng.integers(0, 60, size=cnt)
    secs = rng.integers(0, 60, size=cnt)
    for h, m, s in zip(hrs, mins, secs):
        created_at_list.append(day + pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s)))

created_at = pd.Series(pd.to_datetime(created_at_list)).sort_values().reset_index(drop=True)
N = len(created_at)
print(f"Всего обращений сгенерировано: {N:,}")

df = pd.DataFrame({"created_at": created_at})
df["date"] = df["created_at"].dt.date
df["load_factor"] = df["date"].map(load_by_date).astype(float)

# ----------------------------------------------------------------------------
# Шаг 4. Атрибуты обращения
# ----------------------------------------------------------------------------
# Система
df["system"] = rng.choice(SYS_NAMES, size=N, p=SYS_WEIGHT)
df["criticality"] = df["system"].map(SYS_CRIT)
df["is_critical_for_flights"] = df["system"].map(SYS_CRITICAL_FOR_FLIGHTS)

# Категория — по распределению для каждой системы
df["category"] = ""
for sys in SYS_NAMES:
    mask = df["system"] == sys
    k = int(mask.sum())
    if k:
        df.loc[mask, "category"] = rng.choice(CATEGORIES, size=k, p=CAT_BY_SYS[sys])

# Приоритет — по критичности системы
df["priority"] = ""
for crit, probs in PRIORITY_BY_CRIT.items():
    mask = df["criticality"] == crit
    k = int(mask.sum())
    if k:
        df.loc[mask, "priority"] = rng.choice(PRIORITIES, size=k, p=probs)

# Зона — основная зона системы с вероятностью 0.65, иначе случайная
prim_zone = df["system"].map(PRIMARY_ZONE).values
rand_zone = rng.choice(ALL_ZONES, size=N)
use_primary = rng.random(N) < 0.65
df["zone"] = np.where(use_primary, prim_zone, rand_zone)

# Кто обратился — основная роль зоны с вероятностью 0.65
prim_role = pd.Series(df["zone"]).map(ZONE_ROLE).fillna("Администрация аэропорта").values
rand_role = rng.choice(ALL_ROLES, size=N)
df["reporter_role"] = np.where(rng.random(N) < 0.65, prim_role, rand_role)

# Канал — по приоритету
df["channel"] = ""
for prio, probs in CHANNEL_BY_PRIO.items():
    mask = df["priority"] == prio
    k = int(mask.sum())
    if k:
        df.loc[mask, "channel"] = rng.choice(CHANNELS, size=k, p=probs)

# Команда — по категории (с эскалацией P1/P2 на 2-ю линию/вендора)
df["team"] = ""
for cat, (teams, probs) in TEAM_BY_CAT.items():
    mask = df["category"] == cat
    k = int(mask.sum())
    if k:
        df.loc[mask, "team"] = rng.choice(teams, size=k, p=probs)
# Эскалация части высокоприоритетных заявок 1-й линии на 2-ю линию
esc_mask = (df["priority"].isin(["P1", "P2"])) & (df["team"] == "1-я линия поддержки") & (rng.random(N) < 0.45)
df.loc[esc_mask, "team"] = np.where(
    df.loc[esc_mask, "category"] == "Сеть", "2-я линия / Сети", "2-я линия / Серверы и ПО")

# Сотрудник — случайный из команды
df["agent_id"] = df["team"].map(lambda t: rng.choice(team_to_agents[t]))

# ----------------------------------------------------------------------------
# Шаг 5. Временные метрики: реакция, решение, SLA
# ----------------------------------------------------------------------------
CRIT_MULT = {"Критическая": 1.5, "Высокая": 1.2, "Средняя": 1.0, "Низкая": 0.9}
PRIO_RES_FACTOR = {"P1": 0.75, "P2": 0.90, "P3": 1.05, "P4": 1.30}
CAT_BASE_RES = {"Периферия": 40, "Доступы": 55, "ПО": 110, "Сеть": 140, "Оборудование": 170}
TEAM_FACTOR = {"1-я линия поддержки": 1.0, "2-я линия / Сети": 1.1,
               "2-я линия / Серверы и ПО": 1.15, "Полевая служба": 1.2,
               "Вендор/подрядчик": 1.6}

base_res = df["category"].map(CAT_BASE_RES).astype(float).values
crit_m = df["criticality"].map(CRIT_MULT).astype(float).values
prio_m = df["priority"].map(PRIO_RES_FACTOR).astype(float).values
team_m = df["team"].map(TEAM_FACTOR).astype(float).values
noise_res = rng.lognormal(0.0, 0.45, N)
load = df["load_factor"].values

resolution_min = base_res * crit_m * prio_m * team_m * noise_res * (load ** 0.8)
resolution_min = np.maximum(np.round(resolution_min), 3).astype(int)

resp_target = df["priority"].map(RESP_TARGET).astype(float).values
res_target = df["priority"].map(RES_TARGET).astype(float).values
# «Давление очереди» по часу суток: в утренний и вечерний пик реакция медленнее
hour_idx = df["created_at"].dt.hour.values
hour_pressure = (HOUR_WEIGHTS[hour_idx] / HOUR_WEIGHTS.mean()) ** 0.7
first_response_min = resp_target * rng.lognormal(-0.45, 0.52, N) * (load ** 0.5) * hour_pressure
first_response_min = np.maximum(np.round(first_response_min), 1).astype(int)

df["first_response_min"] = first_response_min
df["resolution_min"] = resolution_min
df["sla_response_target_min"] = resp_target.astype(int)
df["sla_resolution_target_min"] = res_target.astype(int)
df["sla_response_met"] = first_response_min <= resp_target
df["sla_resolution_met"] = resolution_min <= res_target
df["sla_breached"] = ~df["sla_resolution_met"]

# Переоткрытие: чаще у ПО, у спешных P1 и у нарушенных SLA
p_reopen = 0.04 + 0.05 * (df["category"] == "ПО") + 0.04 * (df["priority"] == "P1") \
    + 0.03 * df["sla_breached"]
df["reopened"] = rng.random(N) < p_reopen.values
# У переоткрытых заявок итоговое время решения выше
df.loc[df["reopened"], "resolution_min"] = (df.loc[df["reopened"], "resolution_min"] * 1.4).round().astype(int)
df["sla_resolution_met"] = df["resolution_min"] <= df["sla_resolution_target_min"]
df["sla_breached"] = ~df["sla_resolution_met"]

# ----------------------------------------------------------------------------
# Шаг 6. Влияние на рейсы (только критические инциденты P1 на ключевых системах)
# ----------------------------------------------------------------------------
SEVERITY = {
    "Система обработки багажа (BHS)": 1.4,
    "Стойки и киоски регистрации": 1.3,
    "Посадочные гейты": 1.2,
    "Табло вылета и прилёта (FIDS)": 1.1,
    "Серверы и ЦОД": 1.0,
    "Система контроля доступа (СКУД)": 0.8,
    "Принтеры посадочных талонов": 0.6,
}
downtime = np.zeros(N)
delayed = np.zeros(N, dtype=int)
p1_crit = (df["priority"] == "P1") & (df["is_critical_for_flights"])
idx = np.where(p1_crit.values)[0]
for i in idx:
    dt = df["resolution_min"].iloc[i] * rng.uniform(0.4, 0.8)
    downtime[i] = round(dt)
    sev = SEVERITY.get(df["system"].iloc[i], 0.7)
    delayed[i] = rng.poisson(dt / 65.0 * sev)
df["service_downtime_min"] = downtime.astype(int)
df["delayed_flights"] = delayed

# ----------------------------------------------------------------------------
# Шаг 7. Удовлетворённость (CSAT) — заполнена не у всех
# ----------------------------------------------------------------------------
latent = (4.5
          - 1.3 * df["sla_breached"].astype(float)
          - 0.9 * df["reopened"].astype(float)
          - 0.7 * ((df["resolution_min"] / df["sla_resolution_target_min"]) > 0.8).astype(float)
          + 0.2 * (df["first_response_min"] <= df["sla_response_target_min"]).astype(float)
          + rng.normal(0, 0.6, N))
csat = np.clip(np.round(latent), 1, 5)
responded = rng.random(N) < 0.45
csat = np.where(responded, csat, np.nan)
df["csat"] = csat

# ----------------------------------------------------------------------------
# Шаг 8. Статусы и открытый бэклог у свежих заявок
# ----------------------------------------------------------------------------
df["first_response_at"] = df["created_at"] + pd.to_timedelta(df["first_response_min"], unit="m")
df["resolved_at"] = df["created_at"] + pd.to_timedelta(df["resolution_min"], unit="m")
df["status"] = "Закрыт"

# Часть заявок — отменены (нет решения и оценки)
cancel_mask = rng.random(N) < 0.012
df.loc[cancel_mask, ["resolved_at", "resolution_min", "csat"]] = np.nan
df.loc[cancel_mask, ["sla_resolution_met", "sla_breached"]] = False
df.loc[cancel_mask, ["service_downtime_min", "delayed_flights"]] = 0
df.loc[cancel_mask, "status"] = "Отменён"

# Свежие заявки (последние 8 дней периода) частично ещё в работе
age_days = (DATE_END + pd.Timedelta(days=1) - df["created_at"]).dt.total_seconds() / 86400
p_open = np.clip((8 - age_days) / 8 * 0.6, 0, 0.6)
open_mask = (rng.random(N) < p_open.values) & (~cancel_mask)
open_status = rng.choice(["В работе", "Назначен", "Ожидание пользователя"],
                         size=int(open_mask.sum()), p=[0.5, 0.3, 0.2])
df.loc[open_mask, "status"] = open_status
df.loc[open_mask, ["resolved_at", "resolution_min", "csat"]] = np.nan
df.loc[open_mask, ["sla_resolution_met", "sla_breached"]] = False
df.loc[open_mask, ["service_downtime_min", "delayed_flights"]] = 0

# ----------------------------------------------------------------------------
# Шаг 9. Идентификаторы и финальная раскладка колонок
# ----------------------------------------------------------------------------
df = df.sort_values("created_at").reset_index(drop=True)
df.insert(0, "ticket_id", [f"DME-{i:06d}" for i in range(1, N + 1)])

cols = [
    "ticket_id", "created_at", "first_response_at", "resolved_at", "status",
    "priority", "category", "system", "criticality", "is_critical_for_flights",
    "zone", "channel", "reporter_role", "team", "agent_id",
    "first_response_min", "resolution_min",
    "sla_response_target_min", "sla_resolution_target_min",
    "sla_response_met", "sla_resolution_met", "sla_breached",
    "reopened", "service_downtime_min", "delayed_flights", "csat",
]
df = df[cols]

# ----------------------------------------------------------------------------
# Сохранение файлов
# ----------------------------------------------------------------------------
df.to_csv(RAW_DIR / "tickets.csv", index=False, encoding="utf-8")
flights_daily.to_csv(RAW_DIR / "flights_daily.csv", index=False, encoding="utf-8")

sla_policy = pd.DataFrame({
    "priority": PRIORITIES,
    "priority_name": [PRIORITY_NAME[p] for p in PRIORITIES],
    "response_target_min": [RESP_TARGET[p] for p in PRIORITIES],
    "resolution_target_min": [RES_TARGET[p] for p in PRIORITIES],
})
sla_policy.to_csv(REF_DIR / "sla_policy.csv", index=False, encoding="utf-8")

systems_ref = pd.DataFrame({
    "system": SYS_NAMES,
    "criticality": [SYS_CRIT[s] for s in SYS_NAMES],
    "is_critical_for_flights": [SYS_CRITICAL_FOR_FLIGHTS[s] for s in SYS_NAMES],
})
systems_ref.to_csv(REF_DIR / "systems.csv", index=False, encoding="utf-8")
agents_df.to_csv(REF_DIR / "agents.csv", index=False, encoding="utf-8")

# ----------------------------------------------------------------------------
# Краткая сводка для проверки качества данных
# ----------------------------------------------------------------------------
closed = df[df["resolved_at"].notna()]
daily_tickets = df.groupby(df["created_at"].dt.date).size()
flights_by_date = flights_daily.set_index("date")["flights_scheduled"]
common_idx = daily_tickets.index.intersection(flights_by_date.index)
corr = np.corrcoef(
    flights_by_date.reindex(common_idx).values,
    daily_tickets.reindex(common_idx).values)[0, 1]

print("=" * 60)
print("СВОДКА ПО СГЕНЕРИРОВАННЫМ ДАННЫМ")
print("=" * 60)
print(f"Период:                 {df['created_at'].min()} — {df['created_at'].max()}")
print(f"Всего обращений:        {len(df):,}")
print(f"  закрыто/решено:       {len(closed):,}")
print(f"  открыто (бэклог):     {(df['status'].isin(['В работе','Назначен','Ожидание пользователя'])).sum():,}")
print(f"  отменено:             {(df['status']=='Отменён').sum():,}")
print(f"Нарушение SLA по решению (закрытые): {closed['sla_breached'].mean()*100:.1f}%")
print(f"Нарушение SLA по реакции (закрытые):  {(~closed['sla_response_met']).mean()*100:.1f}%")
print("Нарушение SLA по приоритетам:")
print((closed.groupby('priority')['sla_breached'].mean()*100).round(1).to_string())
print(f"Медианное время решения:  {closed['resolution_min'].median():.0f} мин")
print(f"Доля P1:                {(df['priority']=='P1').mean()*100:.1f}%")
print(f"Задержано рейсов (сумма): {int(df['delayed_flights'].sum()):,}")
print(f"Заполнен CSAT:          {df['csat'].notna().mean()*100:.1f}%  (средний {df['csat'].mean():.2f})")
print(f"Корреляция рейсы↔обращения (по дням): {corr:.2f}")
print("-" * 60)
print("Топ систем по числу обращений:")
print(df["system"].value_counts().head(6).to_string())
print("-" * 60)
print("Распределение по категориям:")
print((df["category"].value_counts(normalize=True) * 100).round(1).to_string())
print("=" * 60)
print("Файлы сохранены в data/raw и data/reference.")
