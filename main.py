"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios
    python main.py --question Q2_quadratic --sweep   # also sweep c_Q (Question 2.(c).iv)

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, DisutilityConsumer,QuadraticDisutilityConsumer, MinimumEnergyConsumer, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule, plot_price_ladder
from src.scenarios import scale_prices, scale_pv, set_tariffs

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_CLASSES = {
    "Q1_caseA": FlexibleConsumerModel,
    "Q1_caseB": FlexibleConsumerModel,
    "Q2_linear": DisutilityConsumer,
    "Q2_quadratic": QuadraticDisutilityConsumer,
    "Q3": MinimumEnergyConsumer,
}


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    # Chooses a different class depending on the question chosen. 
    model_class = MODEL_CLASSES.get(question)
    if model_class is None:
        raise ValueError(f"No model class chosen. Choose one and re-run.")

    model = model_class(data).build()

    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    plot_price_ladder(results, data, save_to=out / "price_ladder.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        # TODO: Determine if this needs to change to implement classes dymamically as in run_base_case
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs

def run_cl_sweep(out: Path, values=(0.0, 0.5, 1.40, 1.41, 1.43, 2.0, 3.1)) -> pd.DataFrame:
    """Question 2.(b).iv: solve Q2_linear for each linear disutility coefficient c_L (DKK/kWh)."""
    base = load_question("Q2_linear")
    rows = []
    for c_L in values:
        data = replace(base, linear_disutility=c_L)
        res = DisutilityConsumer(data).build().solve()
        L, ref = res.hourly["load"], res.hourly["reference_load"]
        dev = L - ref
        at_bound = ((L - data.load_min_kWh).abs() < 1e-6) | ((L - data.load_max_kWh).abs() < 1e-6)
        rows.append({
            "c_L": c_L,
            "procurement_cost": res.meta["economics"]["procurement_cost"],   
            "disutility": (c_L * dev.abs()).sum(),                              
            "energy": L.sum(),                                               
            "abs_deviation": dev.abs().sum(),                                
            "hours_binding": int(at_bound.sum()),                            
        })
        
    table = pd.DataFrame(rows).set_index("c_L")
    table.to_csv(out / "sweep_cL.csv")
    print(table.round(3).to_string())
    return table


def run_cq_sweep(out: Path, values=(0.1, 0.2, 0.5, 1, 2, 5, 10)) -> pd.DataFrame:
    """Question 2.(c).iv: solve Q2_quadratic for each quadratic disutility coefficient c_Q (DKK/kWh^2).

    Returns one row per c_Q with the daily  cost procurementand total disutility (DKK), the daily
    energy consumed and total absolute deviation from the reference (kWh), and the number of hours
    in which the load sits at one of its bounds L_min or L_max. Saves the table as CSV.
    """
    base = load_question("Q2_quadratic")
    rows = []
    for c_Q in values:
        data = replace(base, quadratic_disutility=c_Q)
        res = QuadraticDisutilityConsumer(data).build().solve()
        L, ref = res.hourly["load"], res.hourly["reference_load"]
        dev = L - ref
        at_bound = ((L - data.load_min_kWh).abs() < 1e-6) | ((L - data.load_max_kWh).abs() < 1e-6)
        rows.append({
            "c_Q": c_Q,
            "procurement_cost": res.meta["economics"]["procurement_cost"],   
            "disutility": (c_Q * dev**2).sum(),                              
            "energy": L.sum(),                                               
            "abs_deviation": dev.abs().sum(),                                
            "hours_binding": int(at_bound.sum()),                            
        })
    table = pd.DataFrame(rows).set_index("c_Q")
    table.to_csv(out / "sweep_cQ.csv")
    print(table.round(3).to_string())
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--sweep", action="store_true", help="also run the c_L/c_Q sweep (Q2_linear and Q2_quadratic only)")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    
    if args.sweep:
        if args.question == "Q2_quadratic":
            run_cq_sweep(out)
        elif args.question == "Q2_linear":
            run_cl_sweep(out)
        else:
            print(f"\n[info] --sweep has no effect for {args.question}.")
            
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
