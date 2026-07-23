# -*- coding: utf-8 -*-
"""
Модуль анализа данных IT-поддержки аэропорта.

Отвечает за три задачи:
  1) загрузку и обогащение данных производными признаками;
  2) построение всех графиков отчёта (reports/figures/*.png);
  3) расчёт ключевых метрик и агрегатов (reports/metrics.json,
     dashboard/dashboard_data.json) для README и интерактивной панели.

Запуск:
    python3 src/analysis.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ----------------------------------------------------------------------------
# Пути и оформление
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"
FIG = ROOT / "reports" / "figures"
DASH = ROOT / "dashboard"
FIG.mkdir(parents=True, exist_ok=True)
DASH.mkdir(parents=True, exist_ok=True)

# Единая палитра проекта
NAVY = "#1f3a5f"
STEEL = "#2f6f9f"
SKY = "#6ba9d6"
ORANGE = "#e8833a"
RED = "#c0392b"
GREEN = "#2e8b6f"
GREY = "#8a9099"
SEQ = [NAVY, STEEL, SKY, ORANGE, GREEN, "#9b59b6", "#c0392b", GREY]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.unicode_minus": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#cfd6dd",
    "axes.grid": True,
    "grid.color": "#e8ecf0",
    "grid.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})

WD_ORDER = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
OPEN_STATUSES = ["В работе", "Назначен", "Ожидание пользователя"]


# ----------------------------------------------------------------------------
# Загрузка и обогащение
# ----------------------------------------------------------------------------
def load_data():
    """Читает исходные CSV и возвращает обогащённые датафреймы."""
    tickets = pd.read_csv(
        RAW / "tickets.csv",
        parse_dates=["created_at", "first_response_at", "resolved_at"],
    )
    flights = pd.read_csv(RAW / "flights_daily.csv", parse_dates=["date"])
    sla = pd.read_csv(REF / "sla_policy.csv")
    systems = pd.read_csv(REF / "systems.csv")
    agents = pd.read_csv(REF / "agents.csv")

    t = tickets.copy()
    t["date"] = t["created_at"].dt.normalize()
    t["hour"] = t["created_at"].dt.hour
    t["weekday"] = t["created_at"].dt.weekday
    t["weekday_name"] = t["weekday"].map(dict(enumerate(WD_ORDER)))
    t["month"] = t["created_at"].dt.month
    t["year"] = t["created_at"].dt.year
    t["ym"] = t["created_at"].dt.to_period("M").astype(str)
    t["resolution_hours"] = t["resolution_min"] / 60.0
    t["is_closed"] = t["resolved_at"].notna()
    t["is_open"] = t["status"].isin(OPEN_STATUSES)
    return t, flights, sla, systems, agents


def savefig(fig, name):
    path = FIG / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  сохранён график: reports/figures/{name}")
    return path


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def fig_monthly_volume(t):
    """Помесячная динамика обращений с разбивкой по годам."""
    m = t.groupby("ym").size()
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [NAVY if x.startswith("2024") else ORANGE for x in m.index]
    ax.bar(range(len(m)), m.values, color=colors)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels([x[2:] for x in m.index], rotation=45, ha="right", fontsize=9)
    ax.set_title("Динамика числа обращений по месяцам")
    ax.set_ylabel("Обращений в месяц")
    ax.set_xlabel("Месяц")
    avg24 = m[[i.startswith("2024") for i in m.index]].mean()
    avg25 = m[[i.startswith("2025") for i in m.index]].mean()
    ax.axhline(avg24, color=NAVY, ls="--", lw=1, alpha=0.7)
    ax.axhline(avg25, color=ORANGE, ls="--", lw=1, alpha=0.7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=NAVY, label=f"2024 (сред. {avg24:.0f}/мес)"),
                       Patch(color=ORANGE, label=f"2025 (сред. {avg25:.0f}/мес)")],
              loc="upper left", frameon=False)
    return savefig(fig, "01_monthly_volume.png")


def fig_hour_weekday_heatmap(t):
    """Тепловая карта нагрузки: среднее число обращений по часу и дню недели."""
    n_weeks = t["date"].dt.isocalendar().week.nunique() * t["year"].nunique() / 1.0
    piv = t.pivot_table(index="weekday_name", columns="hour",
                        values="ticket_id", aggfunc="count")
    piv = piv.reindex(WD_ORDER)
    # среднее число обращений в этот час-день (на одну такую дату в периоде)
    n_dates_per_wd = t.groupby("weekday_name")["date"].nunique().reindex(WD_ORDER)
    piv_avg = piv.div(n_dates_per_wd, axis=0)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(piv_avg, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Обращений в час (среднее)"},
                linewidths=0.3, linecolor="white")
    ax.set_title("Когда приходят обращения: час × день недели")
    ax.set_xlabel("Час суток")
    ax.set_ylabel("")
    return savefig(fig, "02_hour_weekday_heatmap.png")


def fig_systems_pareto(t):
    """Парето по системам: где сосредоточена нагрузка."""
    vc = t["system"].value_counts()
    cum = vc.cumsum() / vc.sum() * 100
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.bar(range(len(vc)), vc.values, color=STEEL)
    ax1.set_xticks(range(len(vc)))
    ax1.set_xticklabels(vc.index, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Число обращений", color=STEEL)
    ax1.set_title("Парето по системам аэропорта")
    ax2 = ax1.twinx()
    ax2.plot(range(len(vc)), cum.values, color=ORANGE, marker="o", lw=2)
    ax2.axhline(80, color=RED, ls="--", lw=1, alpha=0.7)
    ax2.set_ylabel("Накопленная доля, %", color=ORANGE)
    ax2.set_ylim(0, 105)
    ax2.grid(False)
    n80 = int((cum <= 80).sum()) + 1
    ax2.annotate(f"{n80} систем дают 80% нагрузки",
                 xy=(n80 - 1, 80), xytext=(n80 + 0.5, 55),
                 arrowprops=dict(arrowstyle="->", color=RED), color=RED, fontsize=10)
    return savefig(fig, "03_systems_pareto.png")


def fig_category_priority(t):
    """Категории обращений с разбивкой по приоритету (доля критичных)."""
    piv = t.pivot_table(index="category", columns="priority",
                        values="ticket_id", aggfunc="count", fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=True).index]
    piv = piv[["P4", "P3", "P2", "P1"]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"P1": RED, "P2": ORANGE, "P3": SKY, "P4": "#c9d6df"}
    left = np.zeros(len(piv))
    for p in ["P4", "P3", "P2", "P1"]:
        ax.barh(piv.index, piv[p], left=left, color=colors[p], label=f"{p}")
        left += piv[p].values
    ax.set_title("Обращения по категориям и приоритету")
    ax.set_xlabel("Число обращений")
    ax.legend(title="Приоритет", loc="lower right", frameon=False)
    return savefig(fig, "04_category_priority.png")


def fig_sla_compliance(t):
    """Соблюдение SLA по приоритетам: реакция против решения."""
    c = t[t["is_closed"]]
    resp = c.groupby("priority")["sla_response_met"].mean() * 100
    res = c.groupby("priority")["sla_resolution_met"].mean() * 100
    order = ["P1", "P2", "P3", "P4"]
    resp = resp.reindex(order)
    res = res.reindex(order)
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w / 2, resp.values, w, color=SKY, label="Реакция в срок")
    b2 = ax.bar(x + w / 2, res.values, w, color=NAVY, label="Решение в срок")
    ax.axhline(90, color=RED, ls="--", lw=1, alpha=0.7, label="Целевой уровень 90%")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Доля в срок, %")
    ax.set_ylim(0, 105)
    ax.set_title("Соблюдение SLA по приоритетам")
    ax.legend(frameon=False, loc="lower right")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", fontsize=9)
    return savefig(fig, "05_sla_compliance_by_priority.png")


def fig_breach_vs_load(t):
    """Связь дневной нагрузки и доли нарушений SLA по реакции."""
    c = t[t["is_closed"]].copy()
    daily = c.groupby("date").agg(
        volume=("ticket_id", "count"),
        resp_breach=("sla_response_met", lambda s: 1 - s.mean()),
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(daily["volume"], daily["resp_breach"] * 100, s=14,
               color=STEEL, alpha=0.45, edgecolor="none")
    z = np.polyfit(daily["volume"], daily["resp_breach"] * 100, 1)
    xs = np.linspace(daily["volume"].min(), daily["volume"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=RED, lw=2, label="Тренд")
    r = np.corrcoef(daily["volume"], daily["resp_breach"])[0, 1]
    ax.set_title("Чем выше дневная нагрузка, тем чаще срыв реакции")
    ax.set_xlabel("Обращений в день")
    ax.set_ylabel("Доля нарушений SLA по реакции, %")
    ax.legend(frameon=False)
    ax.annotate(f"Корреляция r = {r:.2f}", xy=(0.05, 0.92),
                xycoords="axes fraction", fontsize=11, color=RED, fontweight="bold")
    return savefig(fig, "06_breach_vs_load.png")


def fig_mttr_by_system(t):
    """Медианное время решения по системам (в часах)."""
    c = t[t["is_closed"]]
    med = c.groupby("system")["resolution_hours"].median().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = [RED if v >= 4 else ORANGE if v >= 3 else STEEL for v in med.values]
    ax.barh(med.index, med.values, color=colors)
    ax.set_title("Медианное время решения по системам")
    ax.set_xlabel("Часы")
    for i, v in enumerate(med.values):
        ax.text(v + 0.05, i, f"{v:.1f} ч", va="center", fontsize=9)
    ax.set_xlim(0, med.max() * 1.15)
    return savefig(fig, "07_mttr_by_system.png")


def fig_flights_tickets(t, flights):
    """Связь суточного объёма рейсов и числа обращений."""
    daily = t.groupby("date").size().rename("tickets").reset_index()
    fl = flights.rename(columns={"date": "date"})[["date", "flights_scheduled"]]
    m = daily.merge(fl, on="date", how="inner")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(m["flights_scheduled"], m["tickets"], s=14, color=STEEL,
               alpha=0.45, edgecolor="none")
    z = np.polyfit(m["flights_scheduled"], m["tickets"], 1)
    xs = np.linspace(m["flights_scheduled"].min(), m["flights_scheduled"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=ORANGE, lw=2, label="Линия тренда")
    r = np.corrcoef(m["flights_scheduled"], m["tickets"])[0, 1]
    ax.set_title("Нагрузка на поддержку растёт вместе с числом рейсов")
    ax.set_xlabel("Рейсов в день")
    ax.set_ylabel("Обращений в день")
    ax.legend(frameon=False)
    ax.annotate(f"Корреляция r = {r:.2f}", xy=(0.05, 0.92),
                xycoords="axes fraction", fontsize=11, color=ORANGE, fontweight="bold")
    return savefig(fig, "08_flights_vs_tickets.png")


def fig_csat_drivers(t):
    """Что снижает удовлетворённость: нарушение SLA и переоткрытие."""
    c = t[t["csat"].notna()].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # слева — распределение оценок
    vc = c["csat"].value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color=STEEL)
    axes[0].set_title("Распределение оценок CSAT")
    axes[0].set_xlabel("Оценка (1–5)")
    axes[0].set_ylabel("Число ответов")
    # справа — средний CSAT по факторам
    groups = {
        "SLA соблюдён": c[~c["sla_breached"]]["csat"].mean(),
        "SLA нарушен": c[c["sla_breached"]]["csat"].mean(),
        "Без переоткрытия": c[~c["reopened"]]["csat"].mean(),
        "С переоткрытием": c[c["reopened"]]["csat"].mean(),
    }
    colors = [GREEN, RED, GREEN, RED]
    bars = axes[1].bar(list(groups.keys()), list(groups.values()), color=colors)
    axes[1].set_title("Средний CSAT по факторам")
    axes[1].set_ylabel("Средняя оценка")
    axes[1].set_ylim(0, 5)
    axes[1].tick_params(axis="x", rotation=20)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                     f"{b.get_height():.2f}", ha="center", fontsize=10)
    fig.suptitle("Драйверы удовлетворённости пользователей", fontsize=15, fontweight="bold")
    return savefig(fig, "09_csat_drivers.png")


def fig_flight_impact(t):
    """Влияние IT-инцидентов на рейсы: задержки по системам и по месяцам."""
    imp = t[t["delayed_flights"] > 0]
    by_sys = imp.groupby("system")["delayed_flights"].sum().sort_values()
    by_month = t.groupby("ym")["delayed_flights"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(by_sys.index, by_sys.values, color=RED)
    axes[0].set_title("Задержано рейсов по системам")
    axes[0].set_xlabel("Рейсов (сумма за 2 года)")
    for i, v in enumerate(by_sys.values):
        axes[0].text(v + 3, i, str(int(v)), va="center", fontsize=9)
    axes[1].plot(range(len(by_month)), by_month.values, color=NAVY, marker="o", lw=1.8)
    axes[1].set_xticks(range(len(by_month)))
    axes[1].set_xticklabels([x[2:] for x in by_month.index], rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Задержки рейсов из-за IT по месяцам")
    axes[1].set_ylabel("Рейсов в месяц")
    fig.suptitle("Операционное влияние критических IT-инцидентов", fontsize=15, fontweight="bold")
    return savefig(fig, "10_flight_impact.png")


def fig_response_breach_heatmap(t):
    """Тепловая карта доли нарушений реакции по часу и дню недели (для смен)."""
    c = t[t["is_closed"]].copy()
    piv = c.pivot_table(index="weekday_name", columns="hour",
                        values="sla_response_met",
                        aggfunc=lambda s: (1 - s.mean()) * 100).reindex(WD_ORDER)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(piv, cmap="RdYlGn_r", ax=ax, vmin=0, vmax=60,
                cbar_kws={"label": "Нарушений реакции, %"}, linewidths=0.3, linecolor="white")
    ax.set_title("Где горит первичная реакция: час × день недели")
    ax.set_xlabel("Час суток")
    ax.set_ylabel("")
    return savefig(fig, "11_response_breach_heatmap.png")


# ----------------------------------------------------------------------------
# Метрики и агрегаты
# ----------------------------------------------------------------------------
def compute_metrics(t, flights):
    c = t[t["is_closed"]]
    daily = t.groupby("date").size()
    fl = flights.set_index("date")["flights_scheduled"]
    common = daily.index.intersection(fl.index)
    corr = float(np.corrcoef(fl.reindex(common), daily.reindex(common))[0, 1])

    hourly = t.groupby("hour").size()
    peak_hours = hourly.sort_values(ascending=False).head(4).index.tolist()

    metrics = {
        "period": {"start": str(t["created_at"].min().date()),
                   "end": str(t["created_at"].max().date())},
        "total_tickets": int(len(t)),
        "closed_tickets": int(len(c)),
        "open_backlog": int(t["is_open"].sum()),
        "cancelled": int((t["status"] == "Отменён").sum()),
        "resolution_sla_breach_pct": round(float(c["sla_breached"].mean() * 100), 1),
        "response_sla_breach_pct": round(float((~c["sla_response_met"]).mean() * 100), 1),
        "breach_by_priority": {k: round(float(v * 100), 1)
                               for k, v in c.groupby("priority")["sla_breached"].mean().items()},
        "median_resolution_min": int(c["resolution_min"].median()),
        "median_resolution_hours": round(float(c["resolution_hours"].median()), 1),
        "p1_share_pct": round(float((t["priority"] == "P1").mean() * 100), 1),
        "reopen_rate_pct": round(float(c["reopened"].mean() * 100), 1),
        "total_delayed_flights": int(t["delayed_flights"].sum()),
        "delayed_flights_per_year": int(round(t["delayed_flights"].sum() / 2)),
        "total_downtime_hours": int(round(t["service_downtime_min"].sum() / 60)),
        "csat_response_rate_pct": round(float(t["csat"].notna().mean() * 100), 1),
        "csat_mean": round(float(t["csat"].mean()), 2),
        "csat_mean_breached": round(float(c[c["sla_breached"]]["csat"].mean()), 2),
        "csat_mean_ok": round(float(c[~c["sla_breached"]]["csat"].mean()), 2),
        "flights_tickets_corr": round(corr, 2),
        "top_systems": t["system"].value_counts().head(5).to_dict(),
        "category_share_pct": {k: round(float(v * 100), 1)
                               for k, v in t["category"].value_counts(normalize=True).items()},
        "peak_hours": sorted(peak_hours),
        "avg_tickets_per_day": round(float(daily.mean()), 1),
    }
    with open(ROOT / "reports" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("  сохранены метрики: reports/metrics.json")
    return metrics


def build_dashboard_data(t, flights):
    """Готовит компактные агрегаты для интерактивной панели."""
    c = t[t["is_closed"]]
    order = ["P1", "P2", "P3", "P4"]

    monthly = t.groupby("ym").size()
    monthly_breach = c.groupby("ym")["sla_breached"].mean() * 100
    monthly_csat = t.dropna(subset=["csat"]).groupby("ym")["csat"].mean()
    monthly_delay = t.groupby("ym")["delayed_flights"].sum()

    hourly = t.groupby("hour").size()
    n_dates = t["date"].nunique()

    heat = c.pivot_table(index="weekday_name", columns="hour",
                         values="sla_response_met",
                         aggfunc=lambda s: round((1 - s.mean()) * 100, 1)).reindex(WD_ORDER)

    sys_vc = t["system"].value_counts()
    sys_med = c.groupby("system")["resolution_hours"].median()
    sys_breach = c.groupby("system")["sla_breached"].mean() * 100
    systems = []
    for s in sys_vc.index:
        systems.append({
            "system": s,
            "tickets": int(sys_vc[s]),
            "share_pct": round(float(sys_vc[s] / sys_vc.sum() * 100), 1),
            "median_res_h": round(float(sys_med.get(s, 0)), 1),
            "breach_pct": round(float(sys_breach.get(s, 0)), 1),
        })

    prio = []
    for p in order:
        cp = c[c["priority"] == p]
        prio.append({
            "priority": p,
            "tickets": int((t["priority"] == p).sum()),
            "response_compliance": round(float(cp["sla_response_met"].mean() * 100), 1),
            "resolution_compliance": round(float(cp["sla_resolution_met"].mean() * 100), 1),
            "median_res_h": round(float(cp["resolution_hours"].median()), 1),
        })

    delay_by_sys = t[t["delayed_flights"] > 0].groupby("system")["delayed_flights"].sum().sort_values(ascending=False)

    data = {
        "months": [x for x in monthly.index],
        "monthly_tickets": [int(v) for v in monthly.values],
        "monthly_breach": [round(float(monthly_breach.get(m, 0)), 1) for m in monthly.index],
        "monthly_csat": [round(float(monthly_csat.get(m, 0)), 2) for m in monthly.index],
        "monthly_delayed_flights": [int(monthly_delay.get(m, 0)) for m in monthly.index],
        "hours": list(range(24)),
        "hourly_avg": [round(float(hourly.get(h, 0) / n_dates), 2) for h in range(24)],
        "weekday_order": WD_ORDER,
        "response_breach_heatmap": [[float(heat.loc[wd, h]) if not pd.isna(heat.loc[wd, h]) else 0.0
                                     for h in heat.columns] for wd in WD_ORDER],
        "systems": systems,
        "categories": [{"category": k, "tickets": int(v)}
                       for k, v in t["category"].value_counts().items()],
        "priorities": prio,
        "delay_by_system": [{"system": s, "flights": int(v)} for s, v in delay_by_sys.items()],
        "kpis": compute_metrics(t, flights),
    }
    with open(DASH / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("  сохранены агрегаты панели: dashboard/dashboard_data.json")
    return data


def main():
    print("Загрузка данных...")
    t, flights, sla, systems, agents = load_data()
    print(f"Загружено обращений: {len(t):,}")

    print("Построение графиков...")
    fig_monthly_volume(t)
    fig_hour_weekday_heatmap(t)
    fig_systems_pareto(t)
    fig_category_priority(t)
    fig_sla_compliance(t)
    fig_breach_vs_load(t)
    fig_mttr_by_system(t)
    fig_flights_tickets(t, flights)
    fig_csat_drivers(t)
    fig_flight_impact(t)
    fig_response_breach_heatmap(t)

    print("Расчёт метрик и агрегатов...")
    build_dashboard_data(t, flights)
    print("Готово.")


if __name__ == "__main__":
    # Безоконный режим нужен только при запуске из терминала.
    # При импорте из ноутбука backend остаётся inline, и графики
    # встраиваются в вывод ячеек.
    plt.switch_backend("Agg")
    main()
