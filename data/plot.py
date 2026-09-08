import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path(__file__).with_name("HKGov.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot HK Gov series")
    parser.add_argument(
        "--graph", choices=["critical", "death"], default="critical",
        help="Series to plot: critical hospitalizations or deaths",
    )
    parser.add_argument("--start", default="2020-07-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2022-06-15", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--wave", choices=["4", "5", "custom"], default="custom",
        help="Filter by wave: 4 (Ancestral) or 5 (Omicron). Default is the custom date range.",
    )
    args = parser.parse_args(argv)

    # Override the date range when a wave is selected
    if args.wave == "4":
        start_date, end_date = "2020-11-15", "2021-05-15"
    elif args.wave == "5":
        start_date, end_date = "2022-02-01", "2022-05-01"
    else:
        start_date, end_date = args.start, args.end

    if args.graph == "critical":
        ylabel = "Cases"
        x, y = load_hkgov_critical_series(start_date=start_date, end_date=end_date)
    else:
        ylabel = "Deaths"
        x, y = load_hkgov_dead_data(start_date=start_date, end_date=end_date)

    wave_str = f"Wave {args.wave}" if args.wave != "custom" else "Custom Range"
    title = f"HK Gov {args.graph.capitalize()} - {wave_str} ({start_date} to {end_date})"

    plot_data(x, y, title=title, xlabel="Date", ylabel=ylabel, show=False)

    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"hkgov_{args.graph}_wave_{args.wave}.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")
    plt.show()


def plot_data(x, y, title="Data Plot", xlabel="time", ylabel="count", show=False):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker="o", linestyle="-", color="b")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if show:
        plt.show()


def load_hkgov_series(column, start_date=None, end_date=None):
    """Load a column from the HK Gov CSV as dates and numeric values.

    The dataset uses day-first date strings, and many rows are missing
    entries for some columns, so those are dropped.
    """
    df = pd.read_csv(DEFAULT_CSV)
    df["As of date"] = pd.to_datetime(df["As of date"], dayfirst=True, errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["As of date", column]).sort_values("As of date")

    if start_date is not None:
        df = df[df["As of date"] >= pd.to_datetime(start_date, errors="coerce")]
    if end_date is not None:
        df = df[df["As of date"] <= pd.to_datetime(end_date, errors="coerce")]

    return df["As of date"].to_numpy(), df[column].to_numpy(dtype=float)


def load_hkgov_critical_series(start_date=None, end_date=None):
    """Daily count of hospitalized critical cases."""
    return load_hkgov_series(
        "Number of hospitalised cases in critical condition",
        start_date=start_date,
        end_date=end_date,
    )


def load_hkgov_dead_data(start_date=None, end_date=None):
    """Daily count of reported COVID-19 deaths."""
    return load_hkgov_series(
        "Number of death cases", start_date=start_date, end_date=end_date
    )


if __name__ == "__main__":
    main()