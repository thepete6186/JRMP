import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import os

#curerntly haven't found a reliable source for death rate. 
DEFAULT_CSV = Path(__file__).with_name('HKGov.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot HK Gov series')
    parser.add_argument('--graph', choices=['critical', 'death'], default='critical', help='Which graph to plot: critical hospitalizations or deaths')
    parser.add_argument('--start', default='2020-07-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2022-06-15', help='End date (YYYY-MM-DD)')
    args = parser.parse_args(argv)

    if args.graph == 'critical':
        ylabel = 'Cases'
        x, y = load_hkgov_critical_series(start_date=args.start, end_date=args.end)
    else:
        ylabel = 'Deaths'
        x, y = load_hkgov_dead_data(start_date=args.start, end_date=args.end)
    title = f'HK Gov {args.graph.capitalize()} ({args.start} to {args.end})'
    # draw plot (don't block here), then save, then show the window
    plot_data(x, y, title=title, xlabel='Date', ylabel=ylabel, show=False)

    plots_dir = Path(__file__).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f'hkgov_{args.graph}_cases.png'
    plt.savefig(out_path)
    plt.show()

def plot_data(x, y, title='Data Plot', xlabel='time', ylabel='serious_case/hospitiliztion', show=False):

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if show:
        plt.show()


def load_hkgov_critical_series(start_date=None, end_date=None):
    """
    Load HK Gov serious/critical hospitalization proxy data.

    In this dataset, the relevant field is:
    'Number of hospitalised cases in critical condition'
    """

    CSV_PATH = Path(__file__).with_name('HKGov.csv')

    date_col = 'As of date'
    critical_col = 'Number of hospitalised cases in critical condition'

    df = pd.read_csv(CSV_PATH)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df[critical_col] = pd.to_numeric(df[critical_col], errors='coerce')
    df = df.dropna(subset=[date_col, critical_col]).sort_values(date_col)

    # Apply optional date range filtering
    if start_date is not None:
        start_ts = pd.to_datetime(start_date, dayfirst=True, errors='coerce')
        df = df[df[date_col] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date, dayfirst=True, errors='coerce')
        df = df[df[date_col] <= end_ts]

    return df[date_col].to_numpy(), df[critical_col].to_numpy(dtype=float)



""""
def load_hkgov_dead_data(start_date=None, end_date=None):

    Load HK Gov death data.

    In this dataset, the relevant field is:
    'Number of deaths'

    date_col = 'As of date'
    death_col = 'Number of death cases related to COVID-19'

    CSV_PATH = Path(__file__).with_name('HKGov2.csv')

    df = pd.read_csv(CSV_PATH)
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df[death_col] = pd.to_numeric(df[death_col], errors='coerce')
    df = df.dropna(subset=[date_col, death_col]).sort_values(date_col)

    # Apply optional date range filtering
    if start_date is not None:
        start_ts = pd.to_datetime(start_date, dayfirst=True, errors='coerce')
        df = df[df[date_col] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date, dayfirst=True, errors='coerce')
        df = df[df[date_col] <= end_ts]

    return df[date_col].to_numpy(), df[death_col].to_numpy(dtype=float
    )


if __name__ == '__main__':
    main()


"""