"""
High-level functions for applications of the butterfly backtester. The run_parallelized
method runs each ticker in separate process on a separate core. The process is not sped up
exactly proportionally to the number of workers because a large chunk of the total run-time
is waiting for memory transfer, but it helps quite a lot. Also note that Dask is intentionally
avoided here: I tried various configurations as well as various levels of inclusion of Dask in
the script overall, and the current version is always the fastest.
"""

import pandas as pd
import multiprocessing
from typing import Any, Callable
from backtester import ButterflyBacktester

CALENDAR_DIR: str = "../data/EarningsCalendar.parquet"
DATA_DIR_TEMPLATE: str = "/Volumes/Seagate/Selected_Option_Data/CleanedParquets/{}"

earnings_calendar: pd.DataFrame = pd.read_parquet(CALENDAR_DIR)
earnings_dict: dict[str, dict[pd.Timestamp, bool]] = {ticker: {date: bmo for _, date, bmo in rows.values}
                                                      for ticker, rows in earnings_calendar.groupby('Ticker')}


def run_parallel(
        funcs_to_run: str | list[str],
        workers: int = 10,
        kwargs: dict[str, Any] | list[dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Run all the functions given in funcs_to_run parameter across every ticker and
    every earnings date and return the results in dataframe form.
    Each ticker is run in parallel to reduce run time.

    :param funcs_to_run:
        list of functions to run in backtester class in string form
    :param workers:
        number of processes to use for parallel processing
    :param kwargs:
        dictionary of keyword args to pass to the backtest_by_ticker_template

    :return:
        Dataframe with columns = ['Ticker', 'Earnings Date', *funcs_to_run]. One row
        for each earnings date
    """
    if kwargs is None:
        kwargs = {}

    # start multiprocessing pool
    with multiprocessing.Pool(processes=workers) as pool:
        results: list[pd.DataFrame] = []

        # loop through each ticker and assign the work for the entire ticker to a process
        for ticker in earnings_dict:
            results.append(
                pool.apply_async(  # apply_async prevents blocking
                    backtest_by_ticker_template,  # pass everything to backtest_by_ticker defined below
                    kwds={
                        'ticker': ticker,
                        'earnings_dates': earnings_dict[ticker],
                        'funcs_to_run': funcs_to_run,
                        **kwargs
                    }
                )
            )

        # gather results
        results: list[pd.DataFrame] = [result.get() for result in results]

    # concat and organize dataframe
    out = pd.concat(results, axis=0)
    out.set_index('Earnings Date', drop=True, inplace=True)
    out.sort_index(inplace=True)
    return out


def backtest_by_ticker_template(
        ticker: str,
        earnings_dates: dict[pd.Timestamp, bool],
        funcs_to_run: str | list[str],
        func_kwargs: dict[str, Any] | list[dict[str, Any]] = None,
        backtester_kwargs: dict[str, Any] = None
) -> pd.DataFrame:
    """
    Run all funcs_to_run across all earnings dates for a certain ticker using the backtester class.

    :param ticker:
        The 'certain' ticker
    :param earnings_dates:
        dictionary of {earnings_date: bmo} where bmo is a bool that represents before_market_open
    :param funcs_to_run:
        list of strings of names of methods from backtester class to run to get data from
    :param func_kwargs:
        list of dictionaries with keyword arguments to pass to each func_to_run, order matters
    :param backtester_kwargs:
        dictionary of keyword args to pass to backtester constructor

    :return:
        DataFrame with results for each earnings date and separate columns for each func_to_run
    """
    if type(funcs_to_run) == str:
        funcs_to_run: list[str] = [funcs_to_run]
    if func_kwargs is None:
        func_kwargs: list[dict] = [{} for _ in range(len(funcs_to_run))]
    elif type(func_kwargs) == dict:
        func_kwargs: list[dict] = [func_kwargs]
    if backtester_kwargs is None:
        backtester_kwargs: dict = {}
    if len(func_kwargs) != len(funcs_to_run):
        raise ValueError('Function keyword arguments specified incorrectly')

    # generate column names for each function to run
    pairs: list[list[str]] = [
        [f"{key}={value}" for key, value in func_kwarg.items()]
        for func_kwarg in func_kwargs
    ]
    cols: list[str] = [(funcs_to_run[i] + ', '.join(['', *pairs[i]])).strip() for i in range(len(funcs_to_run))]

    # initialize output dataframe
    out = pd.DataFrame(index=list(earnings_dates.keys()), columns=cols)
    out.index.name = 'Earnings Date'

    # read the parquet with option data for the given ticker
    option_data: pd.DataFrame = pd.read_parquet(DATA_DIR_TEMPLATE.format(ticker))

    # calculate valid trading days here to avoid redundancy
    tdays: pd.DatetimeIndex = pd.DatetimeIndex(option_data['DataDate'].unique())

    # loop through each earnings date and perform the backtest
    for edate in earnings_dates:
        try:
            backtester = ButterflyBacktester(
                option_data=option_data,
                earnings_date=edate,
                before_market_open=earnings_dates[edate],
                trading_days=tdays,
                **backtester_kwargs
            )

        # most of the data errors come from the class constructor, in this case
        # most of the funcs_to_run will fail anyway, so move on to next earnings date
        except IndexError as error:
            if str(error) == "Out of data range":
                continue
            else:
                raise error

        # loop through each func_to_run
        for i in range(len(funcs_to_run)):
            method: Callable = getattr(backtester, funcs_to_run[i])
            try:
                curr_result: Any = method(**func_kwargs[i])

            # sometimes we have data errors in this part as well, but at this point it's better
            # to move on to next func instead of cancelling the whole operation
            except IndexError as error:
                if str(error) == "Out of data range":
                    continue
                else:
                    raise error

            # save results to corresponding row and column
            out.loc[edate][(funcs_to_run[i] + ', '.join(['', *pairs[i]])).strip()] = curr_result

    # save data in desired format
    out.reset_index(inplace=True)
    out['Ticker'] = ticker
    return out


if __name__ == '__main__':

    run_parallel(['pnl', 'vol_changes', 'curve_changes', 'quality_of_fill', 'avg_exit_size_ratio'])

