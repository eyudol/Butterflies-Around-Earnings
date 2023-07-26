"""
Class for backtesting iron butterfly trading around earnings. Initializing the class
automatically gathers most of the needed data for all the research methods to minimize
passing of parameters and simplify the parallel backtesting. Noteable backtesting methods
include 'calc_pnl', 'vol_changes', 'curve_changes' 'quality_of_fill', 'avg_exit_size_ratio'.
See documentation of each for details.
"""
import numpy as np
import pandas as pd
from typing import Callable
import warnings
warnings.filterwarnings("error")


class ButterflyBacktester:

    def __init__(
            self,
            option_data: pd.DataFrame,
            earnings_date: pd.Timestamp,
            trading_days: pd.DatetimeIndex,
            before_market_open: bool,
            holding_period: int = 5,
            wing_width: float = 0.1,
            min_days_expiry: int = 10
    ) -> None:
        """
        Gather and store nearly all important data to be used for any of the backtester
        functions. Sometimes gets a bit too much data if you are only calling a single function,
        but this greatly speeds up operations in other cases.

        :param option_data:
            DataFrame with all historical options data for a particular ticker
        :param earnings_date:
            The date of the earnings call that was shown on BamSEC
        :param trading_days:
            Valid trading days for this ticker. Passed to remove redundancy.
        :param before_market_open:
            Boolean value that marks whether the earnings call was before or after the
            trading day on the given earnings_date
        :param holding_period:
            The length in business days that you want to hold the butterfly. Default is
            set to a week
        :param wing_width:
            The width of the wings to use for the butterfly, using 'StrikeProp' units,
            i.e. % moneyness. 0.1 default indicates that default trade is 90%-100%-110%.s
        :param min_days_expiry:
            The minimum days until expiry from the date of the earnings call to look at
            when selecting the option chain to work with. Uses business days, so the default
            represents 2 weeks.
        """
        # use previously calculated valid trading days for this ticker
        self.tdays: pd.DatetimeIndex = trading_days

        # initialize date variables and adjust them in the fix_edate method
        self.edate: pd.Timestamp = earnings_date
        self.edate_idx: int
        self.__fix_edate(before_market_open=before_market_open)

        # determine what option chain (expiry) I will be looking at
        self.expiry: pd.Timestamp
        self.__get_expiry(data=option_data, min_days_expiry=min_days_expiry)

        # will only ever need data from a single option chain
        self.df: pd.DataFrame = option_data[option_data['ExpirationDate'] == self.expiry].copy()

        # determine the exit date for trading
        self.exit_date: pd.Timestamp
        self.__get_exit_date(holding_period=holding_period)

        # determine price of underlying security at entry and at expiry of the option chain
        self.entry_price: float
        self.final_price: float
        self.__get_underlying_prices(option_data=option_data)

        # initialize for use later
        self.wing_width: float = wing_width

        # initialize contracts dictionary which will store all necessary information about
        # each contract. Then call method to get all this info
        self.contracts: dict[str, dict[str, int | float]] = {
            'OTM Put': {},
            'ATM Put': {},
            'ATM Call': {},
            'OTM Call': {}
        }
        self.__get_all_contract_info()

    def __fix_edate(self, before_market_open: bool) -> None:
        """
        Find the earnings date and its index in the valid trading days
        array. Called only in the class constructor.

        :param before_market_open:
            Whether the initialized earnings date needs to be adjusted to be
            before the market open

        :return:
            Nothing, sets instance variables
        """
        # try to find the earnings date in the valid trading days
        try:
            self.edate_idx: int = np.where(self.tdays == self.edate)[0][0]

        # otherwise, find the closest trading date right before the earnings call
        except IndexError:
            diff: np.ndarray[int] = (self.tdays - self.edate).days

            # always want to look strictly before the call
            temp: np.ndarray[int] = diff[diff < 0]

            # in case there are no dates in the option data earlier than the earnings date
            if len(temp) == 0:
                raise IndexError("Out of data range")

            # locate the index
            self.edate_idx: int = np.argmax(temp)  # depending on your system this might throw a type error
            before_market_open: bool = False  # already adjusted to be before the earnings date

        # adjust earnings date so that it always contains the date with the market close
        # right BEFORE the earnings call
        if before_market_open:
            self.edate_idx -= 1

        self.edate: pd.Timestamp = self.tdays[self.edate_idx]

    def __get_expiry(self, data: pd.DataFrame, min_days_expiry: int) -> None:
        """
        Get the best expiry to work with based on the min_days_expiry parameter.
        Called only in the class constructor.

        :param data:
            option data dataframe that is passed to the class constructor
        :param min_days_expiry:
            Hyperparameter that determines how far away the expiry should be
            relative to the earnings date, expressed in business days

        :return:
            None, sets class instance variables
        """
        # determine the earliest possible day for expiry
        try:
            earliest_date: pd.Timestamp = self.tdays[self.edate_idx + min_days_expiry]
        except IndexError:  # in case this is too close to the end of the time series of the data
            raise IndexError("Out of data range")

        # filter to have days only after that
        df: pd.DataFrame = data[data['DataDate'] == earliest_date]

        # get the row with the minimum DTE
        ser: pd.Series = df.iloc[(df['DTE'].abs().argmin())]

        # set expiry
        self.expiry: pd.Timestamp = ser['ExpirationDate']

    def __get_exit_date(self, holding_period: int) -> None:
        """
        Get the best exit date for our trades after the earnings date using the
        holding_period parameter. Called only in class constructor.

        :param holding_period:
            How long, in business days, to hold the position.

        :return:
            None, sets instance variables
        """
        # make sure that the holding period is a reasonable length
        to_expiry: int = (self.expiry - self.edate).days
        if holding_period > to_expiry:
            holding_period: int = to_expiry - 1

        # use previously calculated values to determine exit date
        self.exit_date: pd.Timestamp = self.tdays[self.edate_idx + holding_period]

    def __get_underlying_prices(self, option_data: pd.DataFrame) -> None:
        """
        Get prices of the underlying stock on the date of entry (to determine intrinsic value)
        and on the date of expiry of the selection option chain (to determine value at expiry).
        Called in class constructor

        :param option_data:
            The entire option_data dataframe from the parquets
        :return:
            None, sets instance variables
        """
        # try to find these prices in the isolated dataframes (faster this way)
        try:
            self.final_price: float = self.df[self.df['DataDate'] == self.expiry]['UnderlyingPrice'].iloc[0]
            self.entry_price: float = self.df[self.df['DataDate'] == self.edate]['UnderlyingPrice'].iloc[0]

        # if unable to find them there, search the larger option dataframe
        except IndexError:
            try:
                self.final_price: float = option_data[option_data['DataDate'] == self.expiry]['UnderlyingPrice'].iloc[0]
                self.entry_price: float = option_data[option_data['DataDate'] == self.edate]['UnderlyingPrice'].iloc[0]
            except IndexError:
                raise IndexError("Out of data range")

    def __get_all_contract_info(self) -> None:
        """
        For each of the four options that I trade in the iron butterfly, get and store
        all relevant information about them that might be used in various backtester
        functions. This sometimes get extra unnecessary information but gathering all
        of it at once can save a lot of time in certain cases.

        :return:
            None, stores info to be used later.
        """

        # helper function for calculating intrinsic value of an option at a certain time
        def intrinsic_value(
                row: pd.Series,
                flag: str,
        ) -> float:
            underlying: float = row['UnderlyingPrice']
            strike: float = row['StrikePrice']
            if flag == 'call':
                return max(0.0, underlying - strike)
            else:
                return max(0.0, strike - underlying)

        # helper function for calculating option value at expiration
        def expiration_value(
                strike: float,
                flag: str
        ) -> float:
            if flag == 'call':
                return max(0.0, self.final_price - strike)
            else:
                return max(0.0, strike - self.final_price)

        # helper function that performs all the necessary calculations and storage for a single contract
        def get_single_contract_info(
                contract_name: str,
                strike_prop: float,
        ) -> None:
            # determine whether a put or call and whether buying or selling
            flag: str = contract_name.split()[1].lower()
            wing: bool = True if contract_name[0] == 'O' else False
            df: pd.DataFrame = self.df[self.df['PutCall'] == flag]

            # get the ideal option and entry date to use later
            df_entry: pd.DataFrame = df[df['DataDate'] == self.edate]
            if len(df_entry) == 0:  # this occurs if option was not quoted long enough or there is a data error
                raise IndexError('Out of data range')
            entry: pd.Series = df_entry.iloc[(df_entry['StrikeProp'] - strike_prop).abs().argmin()]

            # find the same option on the exit date
            exit_: pd.DataFrame = df[
                (df['DataDate'] == self.exit_date) & (df['StrikePrice'] == entry['StrikePrice'])
                ]
            if len(exit_) > 1:  # sometimes there is more than one data entry for the same option?
                assert (exit_ == exit_.iloc[0]).all(axis=1).all()
            elif len(exit_) == 0:  # this occurs only if there is a data error
                raise IndexError('Out of data range')
            exit_: pd.Series = exit_.iloc[0]

            # use helper functions to get values to use for calcs below
            entry_intrinsic: float = intrinsic_value(row=entry, flag=flag)
            exit_intrinsic: float = intrinsic_value(row=exit_, flag=flag)
            final_value: float = expiration_value(strike=entry['StrikePrice'], flag=flag)

            # store all necessary information for this contract under the relevant dictionary location
            self.contracts[contract_name]: dict[str, float] = {
                'Strike': entry['StrikePrice'],
                'Entry Price': entry['3/4'] if wing else max(entry_intrinsic, entry['1/4']),
                # option theory specifies a lower theoretical bound
                'Entry Size': entry['AskSize'] if wing else entry['BidSize'],
                'Entry IV': entry['ImpliedVolatility'],
                'Entry Vega': entry['Vega'],
                'Entry Intrinsic': entry_intrinsic,
                'Exit Price': max(exit_intrinsic, exit_['1/4']) if wing else exit_['3/4'],  # minimum theoretical value
                'Exit Size': exit_['BidSize'] if wing else exit_['AskSize'],
                'Exit IV': exit_['ImpliedVolatility'],
                'Exit Vega': exit_['Vega'],
                'Exit Intrinsic': exit_intrinsic,
                'Expiration Value': final_value
            }

        # set ideal strikes based on wing_width parameter
        strike_prop_map: dict[str, float] = {
            'OTM Put': 1 - self.wing_width,
            'ATM Put': 1,
            'ATM Call': 1,
            'OTM Call': 1 + self.wing_width
        }

        # 1/4 represents the price that we can sell at, 3/4 represents the price that we can
        # buy at. Fully crossing the spread on each leg of the trade is rarely profitable.
        self.df['1/4']: pd.Series = (self.df['BidPrice'] + self.df['MidPrice']) / 2
        self.df['3/4']: pd.Series = (self.df['MidPrice'] + self.df['AskPrice']) / 2

        # perform necessary calculations for each contract
        for contract in self.contracts:
            get_single_contract_info(
                contract_name=contract,
                strike_prop=strike_prop_map[contract]
            )

    @staticmethod
    def __curve_calc_disc(df: pd.DataFrame) -> float:
        """
        Calculates curvature of volatility smile of a particular option chain on a particular date
        using a discrete calculation, taking ratio of IV between
        avg(90% moneyness put + 110% moneyness call) / avg(100% moneyness put and call).

        :param df:
            DataFrame that contains only options listed for a particular ticker on
            a particular day for a particular expiry

        :return:
            The smile curvature measure
        """
        # get ATM vol using Deltas instead of % moneyness because in the
        # case of two conflicting options it usually returns the more active chain (higher IV)
        if len(df) == 0:  # fed dataframe is empty
            raise IndexError('Out of data range')
        atm_put: pd.Series = df.iloc[(df['Delta'] + 0.5).abs().argmin()]
        atm_put_iv: float = float(atm_put['ImpliedVolatility'])
        atm_call: pd.Series = df.iloc[(df['Delta'] - 0.5).abs().argmin()]
        atm_call_iv: float = float(atm_call['ImpliedVolatility'])
        atm_vol: float = (atm_put_iv + atm_call_iv) / 2

        # get OTM vol using 90% moneyness put and 110% moneyness call
        df_puts: pd.DataFrame = df[df['PutCall'] == 'put']
        df_calls: pd.DataFrame = df[df['PutCall'] == 'call']
        if len(df_puts) == 0 or len(df_calls) == 0:  # one or the other was empty, doesn't work by definition
            raise IndexError('Out of data range')
        otm_put: pd.Series = df_puts.iloc[(df_puts['StrikeProp'] - 0.9).abs().argmin()]
        otm_put_iv: float = float(otm_put['ImpliedVolatility'])
        otm_call: pd.Series = df_calls.iloc[(df_calls['StrikeProp'] - 1.1).abs().argmin()]
        otm_call_iv: float = float(otm_call['ImpliedVolatility'])
        otm_vol: float = (otm_put_iv + otm_call_iv) / 2

        # curvature is just the ratio between them
        curvature: float = otm_vol / atm_vol

        return curvature

    @staticmethod
    def __curve_calc_cont(df) -> float:
        """
        Calculates curvature of vol smile of a particular option chain on a particular date using
        a continuous measure, specifically by measuring the value of the quadratic term when fitting a
        quadratic regression to the volatility surface on that day. Uses log(strike) as independent variable.

        :param df:
            DataFrame that contains only options listed for a particular ticker on
            a particular day for a particular expiry

        :return:
            The smile curvature measure = the quadratic term
        """
        # I'm trading *iron* butterflies so separating into put side and call side vol
        # surface just makes sense, but also this is standard practice
        put_side: pd.DataFrame = df.loc[(df.PutCall == 'put') & (df.StrikeProp <= 1)]
        call_side: pd.DataFrame = df.loc[(df.PutCall == 'call') & (df.StrikeProp >= 1)]

        if len(put_side) == 0 or len(call_side) == 0:  # need to have some in both
            raise IndexError('Out of data range')
        surface_df: pd.DataFrame = pd.concat([put_side, call_side], axis=0)

        # take logs of strike because call side raw values tend to have larger tails (nature of stocks)
        surface_df['LogStrikeProp']: pd.Series = np.log(surface_df['StrikeProp'])
        try:  # perform the quadratic regression
            quadratic_term: float = np.poly1d(
                np.polyfit(surface_df['LogStrikeProp'], surface_df['ImpliedVolatility'], 2)
            )[2]
        except np.RankWarning:  # rank warnings indicate too few data points, don't want to have weird results
            raise IndexError('Out of data range')

        return quadratic_term

    def curve_changes(
            self,
            method: str = 'continuous',
            days_before: int = 5,
            days_after: int = 5,
    ) -> dict[str, float]:
        """
        Calculate change in the curvature of the smile of the option chain with expiry determined in
        __init__ method from {days_before} business days  before earnings to the day before earnings, and
        from the day before earnings to {days_after} business days after earnings.

        :param method:
            Any variation of the keywords 'discrete' and 'continuous' which indicate
            which curvature calculation method should be used
        :param days_before:
            How many business days to look into the past for the calc for
            curvature change leading up to earnings
        :param days_after:
            How many business days to look into the future for the calc
            for curvature change after earnings

        :return change_before:
            Change in smile curvature leading up to earnings
        :return change_after:
            Change in smile curvature after earnings
        """
        # determine the curvature calculation method
        if 'disc' in method:
            method: Callable = self.__curve_calc_disc
        elif 'cont' in method:
            method: Callable = self.__curve_calc_cont
        else:
            raise ValueError('Invalid curve calculation method')

        # for the curve calcs, never want to use anything with very low liquidity
        df = self.df
        df: pd.DataFrame = df[(df['BidPrice'] > 0)
                              & (df['AskPrice'] > 0)
                              & (df['BidSize'] > 0)
                              & (df['AskSize'] > 0)]

        # get the dates of a week before and after earnings call
        before_date: pd.Timestamp = self.tdays[self.edate_idx - days_before]
        after_date: pd.Timestamp = self.tdays[self.edate_idx + days_after]

        # isolate dataframes to have just the selected dates
        df_before: pd.DataFrame = df[df['DataDate'] == before_date]
        df_after: pd.DataFrame = df[df['DataDate'] == after_date]
        df_earnings: pd.DataFrame = df[df['DataDate'] == self.edate]

        # pass the option chain on the selected date to the curve calc method
        before_curve: float = method(df_before)
        after_curve: float = method(df_after)
        earnings_curve: float = method(df_earnings)

        # calculate the changes in curvature
        change_before: float = earnings_curve - before_curve
        change_after: float = after_curve - earnings_curve

        return {'before': change_before, 'after': change_after}

    def vol_changes(self) -> float:
        """
        Calculate implied volatility changes but now looking at fixed strikes as well as fixed expiries.
        Want to determine if trading the butterflies should be hypothetically profitable, so changes in
        the OTM and ATM vol should be scaled by the average vega over holding period to better reflect
        effect on price. This is far from accurate and has many complexities but gives a general idea.

        :return:
            The vega-scaled difference between change in OTM vol and change in ATM vol.
            A positive value indicates that OTM vol increased relative to OTM vol, e.g. that
            fixed-strike vol-surface became steeper.
        """

        # helper function to avoid redundancy in code, gets entry and exit values for
        # ATM/OTM IV/Vega.
        def get_entry_exit(option_type: str, field: str) -> tuple[float, float]:
            # check input values
            if option_type != 'ATM' and option_type != 'OTM':
                raise ValueError('Invalid option type entry')
            if field != 'IV' and field != 'Vega':
                raise ValueError('Invalid field entry')

            # take the average of put and call values at entry
            entry: float = (self.contracts[f'{option_type} Put'][f'Entry {field}']
                            + self.contracts[f'{option_type} Call'][f'Entry {field}']) / 2

            # take the average of put and call values at exit
            exit_: float = (self.contracts[f'{option_type} Put'][f'Exit {field}']
                            + self.contracts[f'{option_type} Call'][f'Exit {field}']) / 2
            return entry, exit_

        # get ATM IV change
        atm_vol_entry, atm_vol_exit = get_entry_exit('ATM', 'IV')
        atm_vol_change: float = atm_vol_exit - atm_vol_entry

        # get OTM IV change
        otm_vol_entry, otm_vol_exit = get_entry_exit('OTM', 'IV')
        otm_vol_change: float = otm_vol_exit - otm_vol_entry

        # Get average ATM vega
        atm_vega_entry, atm_vega_exit = get_entry_exit('ATM', 'Vega')
        avg_atm_vega: float = (atm_vega_entry + atm_vega_exit) / 2

        # Get average OTM vega
        otm_vega_entry, otm_vega_exit = get_entry_exit('OTM', 'Vega')
        avg_otm_vega: float = (otm_vega_entry + otm_vega_exit) / 2

        # # determine change in ATM Vol
        # atm_vol_entry = (self.contracts['ATM Put']['Entry IV'] + self.contracts['ATM Call']['Entry IV']) / 2
        # atm_vol_exit = (self.contracts['ATM Put']['Exit IV'] + self.contracts['ATM Call']['Exit IV']) / 2
        # atm_vol_change = atm_vol_exit - atm_vol_entry
        #
        # # determine change in OTM Vol
        # otm_vol_entry = (self.contracts['OTM Put']['Entry IV'] + self.contracts['OTM Call']['Entry IV']) / 2
        # otm_vol_exit = (self.contracts['OTM Put']['Exit IV'] + self.contracts['OTM Call']['Exit IV']) / 2
        # otm_vol_change = otm_vol_exit - otm_vol_entry
        #
        # # determine the average vega
        # atm_vega_entry = (self.contracts['ATM Put']['Entry Vega'] + self.contracts['ATM Call']['Entry Vega']) / 2
        # atm_vega_exit = (self.contracts['ATM Put']['Exit Vega'] + self.contracts['ATM Call']['Exit Vega']) / 2
        # avg_atm_vega = (atm_vega_entry + atm_vega_exit) / 2
        #
        # otm_vega_entry = (self.contracts['OTM Put']['Entry Vega'] + self.contracts['OTM Call']['Entry Vega']) / 2
        # otm_vega_exit = (self.contracts['OTM Put']['Exit Vega'] + self.contracts['OTM Call']['Exit Vega']) / 2
        # avg_otm_vega = (otm_vega_entry + otm_vega_exit) / 2

        # calculate the scaled difference
        scaled_diff: float = (otm_vol_change * avg_otm_vega) - (atm_vol_change * avg_atm_vega)
        return scaled_diff * 100  # multiply by 100 to have more reasonable units

    def pnl(self, entry_size_ratio: float = 1) -> float:
        """
        Calculate PnL from trading this butterfly using data fields calculated previously.
        Allows you to specify entry size.

        :param entry_size_ratio:
            The proportion of total available butterfly contracts to actually enter.

        :return:
            The total PnL from this butterfly
        """
        # determine what the size should be, take minimum of available sizes on each leg and
        # multiply by the proportion described above
        actual_entry_size: int = int(
            min([contract['Entry Size'] for contract in self.contracts.values()]) * entry_size_ratio
        )
        if actual_entry_size == 0:
            return 0

        # loop through each leg of the trade and calculate PnL separately
        total_pnl: float = 0
        for name, contract in self.contracts.items():

            # determine if we were long or short
            wing: bool = True if name[0] == 'O' else False

            # determine how many contracts we were able to exit and how many are left over
            remainder: int = max(0, actual_entry_size - contract['Exit Size'])
            offloaded: int = actual_entry_size - remainder

            # for the contracts we offloaded, uses exit prices. For the rest, assume we held to expiry
            main_pnl: float = (contract['Exit Price'] - contract['Entry Price']) * offloaded
            leftover_pnl: float = (contract['Expiration Value'] - contract['Entry Price']) * remainder

            curr_pnl: float = main_pnl + leftover_pnl
            if not wing:
                curr_pnl *= -1

            # add to total pnl
            total_pnl += curr_pnl

        return total_pnl * 100  # multiply by 100 because each option contract represents 100 shares

    def quality_of_fill(self) -> float:
        """
        Calculate a measure the represents whether the fill on this butterfly is favorable:
        uses (net credit / underlying price), which while not extremely representative, does provide
        some insight.

        :return:
            The ratio described above
        """
        # loop through each contract and either add or subtract the price from the net credit
        net_credit: float = 0
        for name, contract in self.contracts.items():
            wing: bool = True if name[0] == 'O' else False
            if wing:
                net_credit -= contract['Entry Price']
            else:
                net_credit += contract['Entry Price']

        # take ratio between net credit and underlying at the time of entry
        return net_credit / self.entry_price

    def avg_exit_size_ratio(self) -> float:
        """
        Calculate the average number of offloaded contracts at the time of exit relative to the
        number of contracts entered. Typically, offloaded contracts represent greater exposure to the
        edge I have in this trade

        :return:
            The average exit size for each of the legs of the trade divided by the entry size
        """
        # get max entry size in the same way as in pnl function
        entry_size: int = min([contract['Entry Size'] for contract in self.contracts.values()])
        if entry_size == 0:
            return np.nan

        # the ratio can be at most 1 because we can't exit more than we entered, otherwise between 0 and 1
        exit_size_ratios: list[float] = [min(1.0, (contract['Exit Size'] / entry_size))
                                         for contract in self.contracts.values()]

        # find the average of the ratios over each leg of the trade
        return sum(exit_size_ratios) / len(exit_size_ratios)
