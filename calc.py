"""
Калькулятор экономики ИИ-решения для Telegram-бота техподдержки.
Все итоговые цифры — в рублях.
Локальный сервер — Selectel Cloud GPU (Россия).
"""

# Цены API-провайдеров в USD за 1M токенов (апрель 2026)
MODELS = [
    {"tier": "Флагман",  "name": "Claude Opus 4.6",  "pin": 5.00,  "pout": 25.00},
    {"tier": "Средний",  "name": "Claude Sonnet 4.6", "pin": 3.00,  "pout": 15.00},
    {"tier": "Бюджет",   "name": "DeepSeek V3.2",     "pin": 0.28,  "pout": 0.42},
]

# Selectel Cloud GPU — актуальные тарифы (апрель 2026, с НДС)
SELECTEL_GPUS = {
    "RTX 4090 (обычная)":    {"hour": 68.54,  "month": 50_037},
    "RTX 4090 (прерываемая)": {"hour": 20.57,  "month": 15_013},
    "Tesla T4 16 ГБ":         {"hour": 10.00,  "month":  7_300},  # ~оценка по рынку
    "A100 80 ГБ":             {"hour": 180.00, "month": 131_400}, # ~оценка по рынку
}
DEFAULT_GPU = "RTX 4090 (прерываемая)"

# Собственный сервер
OWN_CAPEX_RUB  = 280_000   # ₽, RTX 4090 + сервер (оценка 2025)
OWN_HOSTING    = 8_000     # ₽/мес, колокация в ДЦ
OWN_DEVOPS     = 15_000    # ₽/мес, DevOps (частичная занятость)
OWN_MONTHS     = 24        # мес, срок амортизации


def ask(prompt: str, default, cast=float):
    raw = input(f"  {prompt} [{default}]: ").strip()
    return cast(raw) if raw else default


def ask_choice(prompt: str, options: list, default_idx: int = 0) -> int:
    print(f"\n  {prompt}")
    for i, opt in enumerate(options):
        marker = " *" if i == default_idx else "  "
        print(f"  {marker} {i + 1}. {opt}")
    raw = input(f"  Выбор [{default_idx + 1}]: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return int(raw) - 1
    return default_idx


def rub(n: float) -> str:
    if n < 1:
        return "<1 ₽"
    if n < 100:
        return f"{n:.2f} ₽"
    return f"{n:,.0f} ₽"


def divider(char="─", width=58):
    print(char * width)


def header(title: str):
    divider("═")
    print(f"  {title}")
    divider("═")


def section(title: str):
    print()
    divider()
    print(f"  {title}")
    divider()


def main():
    header("Калькулятор экономики ИИ-бота техподдержки (₽)")
    print("\n  Нажмите Enter, чтобы использовать значение по умолчанию.\n")

    usd_rate = ask("Курс USD/₽", 92.0)
    users    = ask("Пользователей в месяц",            500, int)
    rpu      = ask("Запросов на пользователя в день",    2, float)
    tok_in   = ask("Токенов на запрос (in)",            300, int)
    tok_out  = ask("Токенов на ответ (out)",            400, int)

    gpu_names = list(SELECTEL_GPUS.keys())
    default_gpu_idx = gpu_names.index(DEFAULT_GPU)
    gpu_idx  = ask_choice("Выберите GPU-тариф Selectel:", gpu_names, default_gpu_idx)
    gpu_name = gpu_names[gpu_idx]
    gpu_info = SELECTEL_GPUS[gpu_name]

    gpu_hours = ask("Часов работы GPU в сутки (аренда Selectel)", 24, float)
    gpu_devops_rub = ask("DevOps / поддержка при аренде GPU (₽/мес)", 15_000, int)

    req_day = users * rpu
    req_mon = req_day * 30
    m_in    = req_mon * tok_in  / 1_000_000
    m_out   = req_mon * tok_out / 1_000_000

    section("Параметры нагрузки")
    print(f"  Курс USD/₽               : {usd_rate:.0f} ₽")
    print(f"  Запросов в день          : {req_day:,.0f}")
    print(f"  Запросов в месяц         : {req_mon:,.0f}")
    print(f"  Токенов входящих / мес   : {m_in:.2f}M")
    print(f"  Токенов исходящих / мес  : {m_out:.2f}M")

    section("Облачные API-провайдеры (стоимость в месяц)")
    costs_rub = []
    for m in MODELS:
        cin_usd   = m_in  * m["pin"]
        cout_usd  = m_out * m["pout"]
        total_usd = cin_usd + cout_usd
        total_rub = total_usd * usd_rate
        costs_rub.append(total_rub)
        print(f"\n  [{m['tier']}] {m['name']}")
        print(f"    Цена API : ${m['pin']} in / ${m['pout']} out за 1M токенов")
        print(f"    In       : {m_in:.2f}M × ${m['pin']} = ${cin_usd:,.2f}  →  {rub(cin_usd * usd_rate)}")
        print(f"    Out      : {m_out:.2f}M × ${m['pout']} = ${cout_usd:,.2f}  →  {rub(cout_usd * usd_rate)}")
        print(f"    Итого    : ${total_usd:,.2f}  →  {rub(total_rub)} / мес")

    flagship_rub, sonnet_rub, deep_rub = costs_rub

    section(f"Локальное развёртывание — Selectel Cloud GPU")

    gpu_mon_rub   = gpu_info["hour"] * gpu_hours * 30
    gpu_total_rub = gpu_mon_rub + gpu_devops_rub
    print(f"\n  Вариант A — аренда Selectel ({gpu_name}, {gpu_hours:.0f} ч/сут)")
    print(f"    Тариф GPU              : {rub(gpu_info['hour'])}/ч  |  {rub(gpu_info['month'])}/мес (полный)")
    print(f"    Аренда за {gpu_hours:.0f} ч/сут      : {rub(gpu_mon_rub)} / мес")
    print(f"    DevOps / поддержка     : {rub(gpu_devops_rub)} / мес")
    print(f"    Итого                  : {rub(gpu_total_rub)} / мес")
    print(f"    Модель                 : LLaMA 3.1 8B (квантизация Q4)")

    amort_rub = OWN_CAPEX_RUB / OWN_MONTHS
    own_total_rub = amort_rub + OWN_HOSTING + OWN_DEVOPS
    print(f"\n  Вариант B — собственный сервер")
    print(f"    Капитальные затраты    : {rub(OWN_CAPEX_RUB)}  (RTX 4090 + сервер)")
    print(f"    Амортизация ({OWN_MONTHS} мес)    : {rub(amort_rub)} / мес")
    print(f"    Колокация в ДЦ         : {rub(OWN_HOSTING)} / мес")
    print(f"    DevOps                 : {rub(OWN_DEVOPS)} / мес")
    print(f"    Итого                  : {rub(own_total_rub)} / мес")

    section("Точка безубыточности (сервер B vs облако)")
    hybrid_rub = sonnet_rub * 0.3 + deep_rub * 0.7

    def be(cloud_rub: float, label: str) -> str:
        if own_total_rub >= cloud_rub:
            return f"не окупается — сервер ({rub(own_total_rub)}/мес) дороже {label}"
        months = OWN_CAPEX_RUB / (cloud_rub - own_total_rub)
        return f"~{months:.0f} мес"

    print(f"\n  Гибрид 70/30 (DeepSeek/Sonnet)  : {rub(hybrid_rub)} / мес")
    print(f"\n  Сервер vs Opus 4.6               : {be(flagship_rub, 'Opus')}")
    print(f"  Сервер vs Sonnet 4.6             : {be(sonnet_rub, 'Sonnet')}")
    print(f"  Сервер vs Гибрид                 : {be(hybrid_rub, 'Гибрида')}")
    print(f"  Сервер vs DeepSeek V3.2          : {be(deep_rub, 'DeepSeek')}")

    section("Рекомендация")
    if req_day < 500:
        strategy = "Облако — DeepSeek V3.2"
        reason   = (f"При нагрузке {req_day:.0f} зап./день затраты минимальны "
                    f"({rub(deep_rub)}/мес). Никакой инфраструктуры — мгновенный старт.")
    elif req_day < 3000:
        be_hybrid = (OWN_CAPEX_RUB / (hybrid_rub - own_total_rub)
                     if own_total_rub < hybrid_rub else None)
        strategy = "Гибридное облако"
        note     = (f"Selectel-сервер окупится через ~{be_hybrid:.0f} мес."
                    if be_hybrid else "Собственный сервер при таком объёме не окупается.")
        reason   = (f"Простые вопросы (70%) → DeepSeek, сложные (30%) → Sonnet. "
                    f"Итог: {rub(hybrid_rub)}/мес против {rub(sonnet_rub)}/мес. {note}")
    else:
        be_months = (OWN_CAPEX_RUB / (sonnet_rub - own_total_rub)
                     if own_total_rub < sonnet_rub else None)
        if be_months and be_months <= 18:
            strategy = "Гибрид → переход на собственный сервер"
            reason   = (f"Высокая нагрузка ({req_day:.0f} зап./день). Сервер окупается "
                        f"через ~{be_months:.0f} мес. Начать с гибрида, затем перейти на Selectel.")
        else:
            strategy = "Гибридное облако"
            reason   = (f"Высокая нагрузка ({req_day:.0f} зап./день), но собственный сервер "
                        f"({rub(own_total_rub)}/мес + {rub(OWN_CAPEX_RUB)} капитал) "
                        f"не выгоднее гибрида ({rub(hybrid_rub)}/мес).")

    print(f"\n  Стратегия : {strategy}")
    print(f"  Причина   : {reason}")
    print()
    divider("═")
    print()


if __name__ == "__main__":
    main()