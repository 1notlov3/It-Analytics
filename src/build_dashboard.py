# -*- coding: utf-8 -*-
"""
Собирает интерактивную панель dashboard/index.html:
подставляет агрегаты из dashboard/dashboard_data.json в шаблон
dashboard/template.html. Итог — один самодостаточный HTML-файл,
который открывается в любом браузере.

Запуск:
    python3 src/build_dashboard.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"


def main():
    template = (DASH / "template.html").read_text(encoding="utf-8")
    data_json = (DASH / "dashboard_data.json").read_text(encoding="utf-8")
    if "__DATA_JSON__" not in template:
        raise RuntimeError("В шаблоне нет метки __DATA_JSON__")
    html = template.replace("__DATA_JSON__", data_json)
    out = DASH / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Панель собрана: dashboard/index.html ({out.stat().st_size/1024:.0f} КБ)")


if __name__ == "__main__":
    main()
