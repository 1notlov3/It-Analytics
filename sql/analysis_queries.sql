-- ===========================================================================
--  Аналитика IT-поддержки аэропорта Домодедово — ключевые запросы (SQLite)
-- ---------------------------------------------------------------------------
--  Каждый блок помечен строкой  -- @name: <название>  и может быть выполнен
--  как отдельный запрос скриптом src/run_sql.py, который загружает CSV из
--  data/ в базу build/analytics.db и печатает результаты.
--
--  Таблицы:
--    tickets(ticket_id, created_at, resolved_at, status, priority, category,
--            system, criticality, zone, channel, team, agent_id,
--            first_response_min, resolution_min, sla_response_met,
--            sla_resolution_met, sla_breached, reopened,
--            service_downtime_min, delayed_flights, csat, ...)
--    flights_daily(date, flights_scheduled, passengers, ...)
--    sla_policy(priority, priority_name, response_target_min, resolution_target_min)
--    systems(system, criticality, is_critical_for_flights)
--    agents(agent_id, team)
--
--  Булевы поля хранятся как 1/0. Закрытая заявка: resolved_at IS NOT NULL.
-- ===========================================================================


-- @name: KPI — общие показатели службы
SELECT
    (SELECT COUNT(*) FROM tickets)                              AS всего_обращений,
    COUNT(*)                                                   AS закрыто,
    ROUND(100.0 * AVG(CASE WHEN sla_response_met = 0 THEN 1 ELSE 0 END), 1) AS нарушение_реакции_pct,
    ROUND(100.0 * AVG(CASE WHEN sla_breached     = 1 THEN 1 ELSE 0 END), 1) AS нарушение_решения_pct,
    ROUND(AVG(resolution_min) / 60.0, 1)                        AS среднее_время_решения_ч,
    SUM(delayed_flights)                                        AS задержано_рейсов
FROM tickets
WHERE resolved_at IS NOT NULL;


-- @name: Динамика обращений по месяцам
SELECT strftime('%Y-%m', created_at) AS месяц,
       COUNT(*)                      AS обращений
FROM tickets
GROUP BY месяц
ORDER BY месяц;


-- @name: Парето по системам — где сосредоточена нагрузка
SELECT system                                              AS система,
       COUNT(*)                                            AS обращений,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM tickets), 1) AS доля_pct,
       ROUND(100.0 * SUM(COUNT(*)) OVER (ORDER BY COUNT(*) DESC)
             / (SELECT COUNT(*) FROM tickets), 1)          AS накопл_доля_pct
FROM tickets
GROUP BY system
ORDER BY обращений DESC;


-- @name: Категории обращений в разрезе приоритета
SELECT category AS категория,
       SUM(CASE WHEN priority = 'P1' THEN 1 ELSE 0 END) AS P1,
       SUM(CASE WHEN priority = 'P2' THEN 1 ELSE 0 END) AS P2,
       SUM(CASE WHEN priority = 'P3' THEN 1 ELSE 0 END) AS P3,
       SUM(CASE WHEN priority = 'P4' THEN 1 ELSE 0 END) AS P4,
       COUNT(*)                                         AS всего
FROM tickets
GROUP BY category
ORDER BY всего DESC;


-- @name: Соблюдение SLA по приоритетам (реакция и решение)
SELECT t.priority                                                       AS приоритет,
       p.resolution_target_min                                          AS цель_решения_мин,
       COUNT(*)                                                         AS закрыто,
       ROUND(100.0 * AVG(CASE WHEN t.sla_response_met  = 1 THEN 1 ELSE 0 END), 1) AS реакция_в_срок_pct,
       ROUND(100.0 * AVG(CASE WHEN t.sla_resolution_met = 1 THEN 1 ELSE 0 END), 1) AS решение_в_срок_pct,
       ROUND(AVG(t.resolution_min) / 60.0, 1)                           AS среднее_решение_ч
FROM tickets t
JOIN sla_policy p ON p.priority = t.priority
WHERE t.resolved_at IS NOT NULL
GROUP BY t.priority, p.resolution_target_min
ORDER BY t.priority;


-- @name: Среднее время решения по системам
SELECT system                               AS система,
       COUNT(*)                             AS закрыто,
       ROUND(AVG(resolution_min) / 60.0, 2) AS среднее_решение_ч,
       ROUND(100.0 * AVG(CASE WHEN sla_breached = 1 THEN 1 ELSE 0 END), 1) AS нарушение_решения_pct
FROM tickets
WHERE resolved_at IS NOT NULL
GROUP BY system
ORDER BY среднее_решение_ч DESC;


-- @name: Нагрузка и срывы реакции по часам суток (планирование смен)
SELECT CAST(strftime('%H', created_at) AS INTEGER) AS час,
       COUNT(*)                                    AS обращений,
       ROUND(100.0 * AVG(CASE WHEN sla_response_met = 0 THEN 1 ELSE 0 END), 1) AS нарушение_реакции_pct
FROM tickets
WHERE resolved_at IS NOT NULL
GROUP BY час
ORDER BY час;


-- @name: Связь суточного объёма рейсов и числа обращений (JOIN по дням)
WITH daily AS (
    SELECT date(created_at) AS d, COUNT(*) AS tickets
    FROM tickets
    GROUP BY d
)
SELECT f.date                AS дата,
       f.flights_scheduled   AS рейсов,
       d.tickets             AS обращений
FROM flights_daily f
JOIN daily d ON date(d.d) = date(f.date)
ORDER BY f.date
LIMIT 20;


-- @name: Драйверы удовлетворённости (CSAT по факторам)
SELECT ROUND(AVG(CASE WHEN sla_breached = 0 THEN csat END), 2) AS csat_sla_соблюдён,
       ROUND(AVG(CASE WHEN sla_breached = 1 THEN csat END), 2) AS csat_sla_нарушен,
       ROUND(AVG(CASE WHEN reopened     = 0 THEN csat END), 2) AS csat_без_переоткрытия,
       ROUND(AVG(CASE WHEN reopened     = 1 THEN csat END), 2) AS csat_с_переоткрытием
FROM tickets
WHERE csat IS NOT NULL;


-- @name: Операционное влияние на рейсы по системам
SELECT system                                    AS система,
       SUM(delayed_flights)                       AS задержано_рейсов,
       ROUND(SUM(service_downtime_min) / 60.0, 0) AS простой_ч,
       SUM(CASE WHEN priority = 'P1' THEN 1 ELSE 0 END) AS инцидентов_P1
FROM tickets
GROUP BY system
HAVING задержано_рейсов > 0
ORDER BY задержано_рейсов DESC;


-- @name: Команды поддержки — нагрузка, время решения и нарушения
SELECT team                                 AS команда,
       COUNT(*)                             AS закрыто,
       ROUND(AVG(resolution_min) / 60.0, 2) AS среднее_решение_ч,
       ROUND(100.0 * AVG(CASE WHEN sla_breached = 1 THEN 1 ELSE 0 END), 1) AS нарушение_решения_pct
FROM tickets
WHERE resolved_at IS NOT NULL
GROUP BY team
ORDER BY закрыто DESC;
