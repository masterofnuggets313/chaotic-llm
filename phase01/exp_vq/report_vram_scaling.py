"""report_vram_scaling.py — утренняя сводка по VRAM-скейлингу из night_task4.json.

Читает results/night_task4.json (результат Task 4) и формирует фокусный отчёт:
  1. VRAM обучения (окно W=256) STS vs TF на каждом размере лестницы
  2. context-sweep: VRAM по Wc, точка OOM для каждой модели
  3. экстраполяция VRAM на 128K / 1M / 10M (STS ~ O(N), TF ~ O(N^2))
  4. отношение STS/TF VRAM (чем меньше — тем сильнее коммерческий аргумент)

Справочно: почему это важно — attention трансформера O(N^2) по памяти
(KV-cache + матрица внимания), STS-Prog делает top-k выбор по N позициям =>
VRAM растёт ~линейно с контекстом. На длинном контексте квадратичный член
трансформера взрывается, линейный — нет. Это и есть аргумент
"длинный контекст на обычном железе / ноутбуке".

Запуск (утром, после ночного прогона):
  cd phase01\\exp_vq && py -3.13 report_vram_scaling.py
"""
import os
import json
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results", "night_task4.json")
OUT = os.path.join(HERE, "results", "vram_scaling_report.md")
WC_PROJ = [128_000, 1_000_000, 10_000_000]


def fmt_mb(v):
    if v is None:
        return "n/a"
    if v >= 1024:
        return f"{v / 1024:.2f} GB"
    return f"{v:.1f} MB"


def cell(x):
    if isinstance(x, dict) and x.get("oom"):
        return "OOM"
    if isinstance(x, dict) and "vram_mb" in x:
        return fmt_mb(x["vram_mb"])
    if isinstance(x, (int, float)):
        return fmt_mb(x)
    return "n/a"


def main():
    if not os.path.exists(IN):
        print("no data yet:", IN, flush=True)
        sys.exit(0)
    data = json.load(open(IN, encoding="utf-8"))
    targets = sorted({int(k.split("_")[0]) for k in data
                      if k.endswith("_sts") or k.endswith("_tf")})
    if not targets:
        print("T4 ещё не дал результатов.", flush=True)
        sys.exit(0)

    lines = ["# VRAM Scaling Report — STS-Prog vs Transformer", "",
             "_Источник: results/night_task4.json (Task 4, лестница 1M/5M/20M/100M на TinyStories)_", ""]

    # --- Таблица 1: VRAM обучения ---
    lines += ["## 1. VRAM обучения (окно W=256, один прогон)", "",
              "| params | STS VRAM | TF VRAM | STS/TF | STS PPL | TF PPL | STS tok/s | TF tok/s |",
              "|---|---|---|---|---|---|---|---|"]
    for t in targets:
        s = data.get(f"{t}_sts", {})
        tf = data.get(f"{t}_tf", {})
        sv = (s.get("train") or {}).get("train_vram_mb")
        tv = (tf.get("train") or {}).get("train_vram_mb")
        ratio = (sv / tv) if (sv and tv) else None
        lines.append(
            f"| {t:,} | {fmt_mb(sv)} | {fmt_mb(tv)} | {round(ratio, 3) if ratio else 'n/a'} | "
            f"{s.get('ppl')} | {tf.get('ppl')} | {(s.get('train') or {}).get('tok_per_s')} | "
            f"{(tf.get('train') or {}).get('tok_per_s')} |")

    # --- Таблица 2: context-sweep ---
    lines += ["", "## 2. Context-sweep VRAM (свежие модели того же размера, 1 forward B=1)", ""]
    for t in targets:
        s = data.get(f"{t}_sts", {})
        tf = data.get(f"{t}_tf", {})
        lines += [f"### size {t:,}  (STS d={s.get('d')}, TF D={tf.get('D')}, L={s.get('L')})", "",
                  "| Wc | STS VRAM | TF VRAM | STS/TF VRAM |", "|---|---|---|---|"]
        wcs = sorted(set(list((s.get('context_sweep') or {}).keys()) +
                         list((tf.get('context_sweep') or {}).keys())), key=lambda x: int(x))
        for w in wcs:
            sv = (s.get('context_sweep') or {}).get(w)
            tv = (tf.get('context_sweep') or {}).get(w)
            sv_v = sv.get('vram_mb') if isinstance(sv, dict) and 'vram_mb' in sv else sv
            tv_v = tv.get('vram_mb') if isinstance(tv, dict) and 'vram_mb' in tv else tv
            ratio = (sv_v / tv_v) if isinstance(sv_v, (int, float)) and isinstance(tv_v, (int, float)) and tv_v else None
            lines.append(f"| {int(w):,} | {cell(sv)} | {cell(tv)} | {round(ratio, 3) if ratio else 'n/a'} |")
        sp = (s.get('projected_vram_mb') or {})
        tp = (tf.get('projected_vram_mb') or {})
        lines += ["",
                  f"**Экстраполяция:** STS ~{s.get('projection')}, TF ~{tf.get('projection')}"]
        for w in WC_PROJ:
            sw, tw = sp.get(w), tp.get(w)
            ratio = (sw / tw) if (sw and tw) else None
            lines.append(f"  - {w:,} токенов: STS {fmt_mb(sw)} | TF {fmt_mb(tw)} | "
                         f"ratio {round(ratio, 4) if ratio else 'n/a'}")
        lines.append("")

    # --- Резюме ---
    lines += ["## 3. Резюме для утра", ""]
    if targets:
        last = targets[-1]
        s = data.get(f"{last}_sts", {})
        tf = data.get(f"{last}_tf", {})
        sp = (s.get('projected_vram_mb') or {})
        tp = (tf.get('projected_vram_mb') or {})
        lines.append(
            f"- На максимальном размере ({last:,}) тренировочный VRAM: STS {fmt_mb((s.get('train') or {}).get('train_vram_mb'))} "
            f"vs TF {fmt_mb((tf.get('train') or {}).get('train_vram_mb'))}.")
        lines.append(
            f"- Экстраполяция на 1M контекст: STS {fmt_mb(sp.get(1_000_000))} vs TF {fmt_mb(tp.get(1_000_000))} "
            f"(ratio {round(sp.get(1_000_000)/tp.get(1_000_000), 3) if (sp.get(1_000_000) and tp.get(1_000_000)) else 'n/a'}).")
        lines.append(
            f"- Экстраполяция на 10M контекст: STS {fmt_mb(sp.get(10_000_000))} vs TF {fmt_mb(tp.get(10_000_000))} "
            f"(ratio {round(sp.get(10_000_000)/tp.get(10_000_000), 3) if (sp.get(10_000_000) and tp.get(10_000_000)) else 'n/a'}).")
        lines.append("- Если STS держит STS/TF < 1 на ВСЕХ размерах и во всех точках context-sweep — "
                     "это чистый аргумент 'длинный контекст на обычном железе'.")
        lines.append("- ЧЕСТНО: истинный 10M-контекст на ТЕКУЩЕЙ архитектуре требует вынести позиции "
                     "self.pos=(1,W,d) в производную кодировку (будущая инженерия); цифры выше — тренд + "
                     "асимптотическая экстраполяция O(N)/O(N^2), не буквальный прогон на 10M.")

    open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"\nsaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
