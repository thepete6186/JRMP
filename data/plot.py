import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import os

# currently haven't found a reliable source for death rate. 
DEFAULT_CSV = Path(__file__).with_name('HKGov.csv')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot HK Gov series')
    parser.add_argument('--graph', choices=['critical', 'death'], default='critical', help='Which graph to plot: critical hospitalizations or deaths')
    parser.add_argument('--start', default='2020-07-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2022-06-15', help='End date (YYYY-MM-DD)')
    parser.add_argument('--wave', choices=['4', '5', 'custom'], default='custom', help='Filter by wave: 4 (Ancestral) or 5 (Omicron). Default is custom range.') 
    args = parser.parse_args(argv)

    # Override start and end dates based on the selected wave
    if args.wave == '4':
        start_date = '2020-11-15'
        end_date = '2021-05-15'
    elif args.wave == '5':
        start_date = '2022-01-01'
        end_date = '2022-06-01'
    else:
        # Fall back to whatever was provided in --start and --end
        start_date = args.start
        end_date = args.end

    if args.graph == 'critical':
        ylabel = 'Cases'
        x, y = load_hkgov_critical_series(start_date=start_date, end_date=end_date)
    else:
        ylabel = 'Deaths'
        # Adjust this function if it exists or when you implement it
        x, y = load_hkgov_dead_data(start_date=start_date, end_date=end_date)
        
    # Dynamically update the title to reflect the chosen wave or custom dates
    wave_str = f"Wave {args.wave}" if args.wave != 'custom' else "Custom Range"
    title = f'HK Gov {args.graph.capitalize()} - {wave_str} ({start_date} to {end_date})'
    
    # draw plot (don't block here), then save, then show the window
    plot_data(x, y, title=title, xlabel='Date', ylabel=ylabel, show=False)

    plots_dir = Path(__file__).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f'hkgov_{args.graph}_wave_{args.wave}.png'
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
        start_ts = pd.to_datetime(start_date, errors='coerce')
        df = df[df[date_col] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date, errors='coerce')
        df = df[df[date_col] <= end_ts]

    return df[date_col].to_numpy(), df[critical_col].to_numpy(dtype=float)

if __name__ == '__main__':
    main()