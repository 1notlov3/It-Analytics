# -*- coding: utf-8 -*-
"""
Собирает и исполняет три Jupyter-ноутбука проекта. Графики и таблицы
встраиваются в вывод, поэтому ноутбуки читаются прямо на GitHub.

Запуск:
    python3 src/build_notebooks.py
"""

from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

IMPORTS = (
    "%matplotlib inline\n"
    "import sys, warnings\n"
    "from pathlib import Path\n"
    "sys.path.append(str(Path.cwd().parent / 'src'))\n"
    "warnings.filterwarnings('ignore')\n"
    "import numpy as np, pandas as pd\n"
    "import matplotlib.pyplot as plt, seaborn as sns\n"
    "from analysis import load_data, NAVY, STEEL, SKY, ORANGE, RED, GREEN, WD_ORDER\n"
    "pd.set_option('display.max_columns', 30)\n"
    "plt.rcParams.update({'figure.dpi': 110, 'font.size': 11, 'axes.grid': True,\n"
    "                     'grid.color': '#e8ecf0', 'axes.unicode_minus': False})\n"
    "t, flights, sla, systems, agents = load_data()\n"
    "print('Обращений:', len(t), '| период:', t.created_at.min().date(), '—', t.created_at.max().date())\n"
    "t.head(3)"
)

# ---------------------------------------------------------------------------
# Ноутбук 1. Обзор данных, объёмы и сезонность
# ---------------------------------------------------------------------------
nb1 = [
    ("md", "# 01 · Обзор данных IT-поддержки аэропорта\n\n"
           "**Проект:** аналитика службы технической поддержки аэропорта Домодедово.\n\n"
           "**Данные:** синтетические (сгенерированы `data/synthetic_data_generator.py`), но "
           "смоделированы так, чтобы повторять поведение реальной службы: суточные пики под "
           "волны рейсов, недельная и годовая сезонность, старение оборудования, нарушения SLA "
           "под нагрузкой и влияние критических инцидентов на рейсы.\n\n"
           "**Период:** 2 полных года (2024–2025). В этом ноутбуке знакомимся с набором данных, "
           "проверяем качество и смотрим, как распределена нагрузка во времени."),
    ("code", IMPORTS),
    ("md", "## Состав данных\n\n"
           "Главная таблица — журнал обращений `tickets` (одна строка = одно обращение). "
           "Ключевые поля: время создания/реакции/решения, приоритет (P1–P4), категория, "
           "затронутая система, зона аэропорта, канал, команда и сотрудник, метрики SLA, "
           "флаг переоткрытия, простой и число задержанных рейсов, оценка CSAT."),
    ("code",
     "print('Размерность:', t.shape)\n"
     "print('\\nСтатусы обращений:')\n"
     "print(t.status.value_counts())\n"
     "print('\\nПропуски по ключевым полям (ожидаемы у открытых/отменённых и по CSAT):')\n"
     "print(t[['resolved_at','resolution_min','csat']].isna().sum())"),
    ("md", "## Динамика объёма и рост год к году\n\n"
           "Смотрим число обращений по месяцам. Синим — 2024, оранжевым — 2025."),
    ("code",
     "m = t.groupby('ym').size()\n"
     "avg24 = m[[i.startswith('2024') for i in m.index]].mean()\n"
     "avg25 = m[[i.startswith('2025') for i in m.index]].mean()\n"
     "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
     "colors = [NAVY if x.startswith('2024') else ORANGE for x in m.index]\n"
     "ax.bar(range(len(m)), m.values, color=colors)\n"
     "ax.set_xticks(range(len(m))); ax.set_xticklabels([x[2:] for x in m.index], rotation=45, ha='right', fontsize=9)\n"
     "ax.set_title('Число обращений по месяцам'); ax.set_ylabel('обращений/мес')\n"
     "plt.show()\n"
     "print(f'Средний объём: 2024 = {avg24:.0f}/мес, 2025 = {avg25:.0f}/мес, рост {avg25/avg24*100-100:.1f}%')"),
    ("md", "## Когда приходят обращения\n\n"
           "Тепловая карта «час × день недели» (среднее число обращений в час). "
           "Хорошо видны две волны — утренняя и вечерняя, совпадающие с банками рейсов."),
    ("code",
     "piv = t.pivot_table(index='weekday_name', columns='hour', values='ticket_id', aggfunc='count').reindex(WD_ORDER)\n"
     "nd = t.groupby('weekday_name')['date'].nunique().reindex(WD_ORDER)\n"
     "piv_avg = piv.div(nd, axis=0)\n"
     "fig, ax = plt.subplots(figsize=(13, 4))\n"
     "sns.heatmap(piv_avg, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'обращений/час'}, linewidths=.3, linecolor='white')\n"
     "ax.set_title('Нагрузка: час × день недели'); ax.set_xlabel('час суток'); ax.set_ylabel('')\n"
     "plt.show()"),
    ("code",
     "h = t.groupby('hour').size() / t.date.nunique()\n"
     "fig, ax = plt.subplots(figsize=(11, 4))\n"
     "ax.bar(h.index, h.values, color=[ORANGE if x in [7,8,17,18] else STEEL for x in h.index])\n"
     "ax.set_title('Среднее число обращений по часам суток'); ax.set_xlabel('час'); ax.set_ylabel('обращений/час')\n"
     "plt.show()\n"
     "print('Пиковые часы (max нагрузка):', list(h.sort_values(ascending=False).head(4).index))"),
    ("md", "### Выводы блока\n\n"
           "- Нагрузка стабильно растёт год к году вслед за пассажиропотоком.\n"
           "- Обращения приходят двумя волнами — **утро 06:00–09:00** и **вечер 17:00–19:00**.\n"
           "- Будни заметно нагруженнее выходных.\n\n"
           "Дальше (ноутбук 02) разберём, какие системы и категории создают нагрузку и как "
           "служба справляется с SLA."),
]

# ---------------------------------------------------------------------------
# Ноутбук 2. Системы, категории и SLA
# ---------------------------------------------------------------------------
nb2 = [
    ("md", "# 02 · Системы, категории и соблюдение SLA\n\n"
           "Разбираемся, где сосредоточена нагрузка и насколько служба выдерживает целевые "
           "сроки реакции и решения."),
    ("code", IMPORTS),
    ("md", "## Парето по системам\n\n"
           "Принцип 80/20: небольшое число систем даёт основную часть обращений."),
    ("code",
     "vc = t.system.value_counts(); cum = vc.cumsum() / vc.sum() * 100\n"
     "fig, ax1 = plt.subplots(figsize=(11, 5.5))\n"
     "ax1.bar(range(len(vc)), vc.values, color=STEEL)\n"
     "ax1.set_xticks(range(len(vc))); ax1.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=8)\n"
     "ax1.set_ylabel('обращений')\n"
     "ax2 = ax1.twinx(); ax2.plot(range(len(vc)), cum.values, color=ORANGE, marker='o'); ax2.grid(False)\n"
     "ax2.axhline(80, color=RED, ls='--'); ax2.set_ylabel('накопленная доля, %'); ax2.set_ylim(0, 105)\n"
     "ax1.set_title('Парето по системам аэропорта'); plt.show()\n"
     "print(vc.head(8).to_string())"),
    ("md", "## Категории обращений и их критичность\n\n"
           "Стек по приоритетам показывает не только объём категории, но и долю критичных заявок."),
    ("code",
     "piv = t.pivot_table(index='category', columns='priority', values='ticket_id', aggfunc='count', fill_value=0)\n"
     "piv = piv.loc[piv.sum(1).sort_values().index][['P4','P3','P2','P1']]\n"
     "ax = piv.plot(kind='barh', stacked=True, figsize=(10, 4.5), color=['#c9d6df', SKY, ORANGE, RED])\n"
     "ax.set_title('Категории обращений × приоритет'); ax.set_xlabel('обращений'); ax.legend(title='Приоритет')\n"
     "plt.show()"),
    ("md", "## Соблюдение SLA по приоритетам\n\n"
           "Сравниваем два SLA — на **первичную реакцию** и на **решение**. Пунктир — цель 90%."),
    ("code",
     "c = t[t.is_closed]\n"
     "order = ['P1','P2','P3','P4']\n"
     "resp = (c.groupby('priority').sla_response_met.mean()*100).reindex(order)\n"
     "res = (c.groupby('priority').sla_resolution_met.mean()*100).reindex(order)\n"
     "x = np.arange(4); w = 0.38\n"
     "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
     "ax.bar(x-w/2, resp.values, w, color=SKY, label='реакция в срок')\n"
     "ax.bar(x+w/2, res.values, w, color=NAVY, label='решение в срок')\n"
     "ax.axhline(90, color=RED, ls='--', label='цель 90%')\n"
     "ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylim(0, 105); ax.set_ylabel('%'); ax.legend()\n"
     "ax.set_title('Соблюдение SLA по приоритетам'); plt.show()\n"
     "pd.DataFrame({'реакция_в_срок_%': resp.round(1), 'решение_в_срок_%': res.round(1)})"),
    ("md", "## Время решения и нарушения по системам"),
    ("code",
     "med = c.groupby('system').resolution_hours.median().sort_values()\n"
     "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
     "ax.barh(med.index, med.values, color=[RED if v>=4 else ORANGE if v>=3 else STEEL for v in med.values])\n"
     "ax.set_title('Медианное время решения по системам'); ax.set_xlabel('часы'); plt.show()\n"
     "print('Доля нарушений SLA по решению, топ-6 систем:')\n"
     "print((c.groupby('system').sla_breached.mean()*100).sort_values(ascending=False).round(1).head(6).to_string())"),
    ("md", "## Нагрузка дня и срывы реакции\n\n"
           "Проверяем гипотезу: чем выше дневная нагрузка, тем чаще срывается первичная реакция."),
    ("code",
     "d = c.groupby('date').agg(volume=('ticket_id','count'), rb=('sla_response_met', lambda s: 1-s.mean()))\n"
     "r = np.corrcoef(d.volume, d.rb)[0,1]\n"
     "fig, ax = plt.subplots(figsize=(9, 5))\n"
     "ax.scatter(d.volume, d.rb*100, s=12, alpha=.4, color=STEEL)\n"
     "z = np.polyfit(d.volume, d.rb*100, 1); xs = np.linspace(d.volume.min(), d.volume.max(), 50)\n"
     "ax.plot(xs, np.polyval(z, xs), color=RED, lw=2)\n"
     "ax.set_title(f'Нагрузка дня vs срыв реакции (r = {r:.2f})'); ax.set_xlabel('обращений/день'); ax.set_ylabel('срыв реакции, %')\n"
     "plt.show()"),
    ("md", "### Выводы блока\n\n"
           "- **9 из 13 систем** дают ~80% обращений; лидеры — рабочие станции, киоски "
           "регистрации, Wi-Fi, табло FIDS.\n"
           "- SLA на **решение** держится хорошо (кроме P1), а вот **реакция** — узкое место.\n"
           "- Дольше всего чинят **BHS** и **серверы** — там же выше доля нарушений.\n"
           "- Срыв реакции растёт с дневной нагрузкой — это вопрос расстановки смен (ноутбук 03)."),
]

# ---------------------------------------------------------------------------
# Ноутбук 3. Влияние на бизнес и рекомендации
# ---------------------------------------------------------------------------
nb3 = [
    ("md", "# 03 · Влияние на бизнес и рекомендации\n\n"
           "Связываем IT-поддержку с операционной деятельностью аэропорта: нагрузка от рейсов, "
           "удовлетворённость пользователей, влияние инцидентов на рейсы — и формулируем "
           "рекомендации."),
    ("code", IMPORTS),
    ("md", "## Нагрузка на поддержку и объём рейсов\n\n"
           "Строим простую модель зависимости числа обращений от числа рейсов в день."),
    ("code",
     "daily = t.groupby('date').size().rename('tickets').reset_index()\n"
     "mg = daily.merge(flights[['date','flights_scheduled']], on='date')\n"
     "r = np.corrcoef(mg.flights_scheduled, mg.tickets)[0,1]\n"
     "z = np.polyfit(mg.flights_scheduled, mg.tickets, 1)\n"
     "fig, ax = plt.subplots(figsize=(9, 5))\n"
     "ax.scatter(mg.flights_scheduled, mg.tickets, s=12, alpha=.4, color=STEEL)\n"
     "xs = np.linspace(mg.flights_scheduled.min(), mg.flights_scheduled.max(), 50)\n"
     "ax.plot(xs, np.polyval(z, xs), color=ORANGE, lw=2)\n"
     "ax.set_title(f'Рейсы vs обращения (r = {r:.2f})'); ax.set_xlabel('рейсов/день'); ax.set_ylabel('обращений/день')\n"
     "plt.show()\n"
     "print(f'Модель: обращений ≈ {z[0]:.3f} × рейсы + {z[1]:.0f}')\n"
     "print(f'Каждые +100 рейсов в день дают примерно +{z[0]*100:.0f} обращений')"),
    ("md", "## Что снижает удовлетворённость (CSAT)"),
    ("code",
     "cc = t[t.csat.notna()]\n"
     "g = {'SLA соблюдён': cc[~cc.sla_breached].csat.mean(), 'SLA нарушен': cc[cc.sla_breached].csat.mean(),\n"
     "     'Без переоткрытия': cc[~cc.reopened].csat.mean(), 'С переоткрытием': cc[cc.reopened].csat.mean()}\n"
     "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
     "ax.bar(list(g.keys()), list(g.values()), color=[GREEN, RED, GREEN, RED])\n"
     "ax.set_ylim(0, 5); ax.set_title('Средний CSAT по факторам'); ax.tick_params(axis='x', rotation=15)\n"
     "for i, v in enumerate(g.values()): ax.text(i, v+0.05, f'{v:.2f}', ha='center')\n"
     "plt.show()"),
    ("md", "## Операционное влияние критических инцидентов\n\n"
           "Критические инциденты (P1) на ключевых системах приводят к простоям и задержкам рейсов."),
    ("code",
     "imp = t[t.delayed_flights > 0]\n"
     "by = imp.groupby('system').delayed_flights.sum().sort_values()\n"
     "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
     "ax.barh(by.index, by.values, color=RED)\n"
     "ax.set_title('Задержано рейсов по системам (сумма за 2 года)'); plt.show()\n"
     "print('Всего задержано рейсов:', int(t.delayed_flights.sum()), '| в год ≈', int(t.delayed_flights.sum()/2))\n"
     "print('Суммарный простой критичных систем, ч:', int(t.service_downtime_min.sum()/60))"),
    ("md", "## Нагрузка по интервалам суток — основа для смен"),
    ("code",
     "band = pd.cut(t.hour, bins=[-1,5,9,16,19,23],\n"
     "              labels=['Ночь 0–5','Утро 6–9','День 10–16','Вечер 17–19','Поздний 20–23'])\n"
     "load = (t.groupby(band).size() / t.date.nunique()).round(1)\n"
     "print('Средняя нагрузка по интервалам (обращений/день):')\n"
     "print(load.to_string())"),
    ("md", "## Выводы и рекомендации\n\n"
           "**Ключевые находки**\n"
           "1. Первичная **реакция — главное узкое место**: срыв SLA по реакции ~31%, "
           "сконцентрирован в утренний и вечерний пик.\n"
           "2. **Решение** укладывается в срок почти везде, кроме **P1** (критические) и систем "
           "**BHS**/**серверы** — самых сложных.\n"
           "3. Нагрузка на поддержку **растёт вместе с рейсами** (r≈0.5) и год к году.\n"
           "4. Нарушение SLA обваливает CSAT в ~1.8 раза — качество сервиса напрямую зависит от сроков.\n"
           "5. IT-инциденты стали причиной **~655 задержек рейсов в год**.\n\n"
           "**Рекомендации**\n"
           "- **Усилить 1-ю линию в пик** (06:00–09:00 и 17:00–19:00) — прямой способ снять срывы реакции.\n"
           "- **Профилактика по BHS и серверам**: мониторинг и предиктивное обслуживание сократят "
           "долгие P1 и задержки рейсов.\n"
           "- **Обновление парка киосков регистрации и АРМ** — лидеров по числу обращений.\n"
           "- **Планирование ёмкости от расписания рейсов**: модель «рейсы → обращения» позволяет "
           "прогнозировать нагрузку и заранее ставить смены.\n"
           "- **Работа с переоткрытиями**: контроль качества решения повысит CSAT."),
]


def build(cells, path, title):
    nb = new_notebook()
    nb.cells = [new_markdown_cell(c) if kind == "md" else new_code_cell(c) for kind, c in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
    nbf.write(nb, path)
    print(f"  собран и исполнен: notebooks/{path.name}")


def main():
    print("Сборка ноутбуков...")
    build(nb1, NB_DIR / "01_data_overview.ipynb", "Обзор данных")
    build(nb2, NB_DIR / "02_systems_and_sla.ipynb", "Системы и SLA")
    build(nb3, NB_DIR / "03_impact_and_recommendations.ipynb", "Влияние и рекомендации")
    print("Готово.")


if __name__ == "__main__":
    main()
