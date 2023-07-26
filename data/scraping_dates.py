"""
Script for scraping earnings dates from BamSEC using Selenium.
The resulting parquet that gets saved has the following structure:
Columns    |   TICKER: string   |   Earnings Date: datetime64   |   BMO: bool
Row 1      |       'AAPL'       |   pd.Timestamp('2022-06-16')  |    True
Row 2      |       'TSLA'       |   pd.Timestamp('2020-01-13')  |    False
etc.
Includes the top 250 companies by market cap from the S&P 500
'bmo' = before market open (True), 'amc' = after market close (False).
This script requires a BamSEC account.
Note that if BamSEC UI changes, some of this code may not work anymore.
"""
import pandas as pd
import time as tm

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement

import logging

logging.basicConfig(format='%[(levelname)]s: %(message)s')

URL: str = 'https://www.bamsec.com'
USERNAME: str = "BamSEC username"
PASSWORD: str = "BamSEC password"
EARLIEST_YEAR: int = 2018

# need to have webdriver executable in your current path
driver = webdriver.Chrome()

# log in to BamSEC
driver.get(URL)
driver.find_elements(By.XPATH, "//a[contains(.,'Log In')]")[0].click()
driver.find_elements(By.XPATH, "//input[@name='email']")[0].send_keys(USERNAME)
driver.find_elements(By.XPATH, "//input[@name='password']")[0].send_keys(PASSWORD)
driver.find_elements(By.XPATH, "//button[@type='submit']")[0].click()
logging.info('Logged into BamSEC')

# get list of top 250 S&P500 companies
tickers: list[str] = pd.read_excel('Top 250 Companies.xlsx', usecols=['Ticker'], dtype={'Ticker': 'string'})['Ticker'].to_list()

# loop through each ticker and save results
df_list: list[pd.DataFrame] = []
for ticker in tickers:

    # initialize dataframe for current ticker
    df: pd.DataFrame = pd.DataFrame(columns=['Ticker', 'Earnings Date', 'BMO'])

    # go to the home page of the current ticker
    driver.find_elements(By.XPATH, "//input[@type='search']")[0].send_keys(ticker)
    driver.find_elements(By.XPATH, "//button[@type='submit']")[0].click()

    try:  # go to the transcripts tab, for some tickers this tab does not exist, weirdly, so just skip
        driver.find_elements(By.XPATH, "//a[contains(.,'Transcripts')]")[0].click()
    except NoSuchElementException:
        logging.warning(f"Couldn't find 'Transcripts' tab for {ticker}, continuing to next")
        continue

    # left labels contain type of transcript, right labels contain date of transcript
    left_labels: list[WebElement] = driver.find_elements(By.XPATH, "//span[@class='label-left']")
    right_labels: list[WebElement] = driver.find_elements(By.XPATH, "//span[@class='label-right']")

    # go through each earnings transcript and save the date and time
    dates: list[str] = []
    bmos: list[bool] = []
    logging.debug(f"Started loop for {ticker}")
    for i in range(len(left_labels)):

        # has to be an earnings transcript
        if "Earnings" in left_labels[i].text:

            # if we've gone too far back in time then we are done
            if int(right_labels[i].text[6:8]) < EARLIEST_YEAR - 2000:
                break

            # save the date
            dates.append(right_labels[i].text)

            # actually go look at the transcript and extract the time, which is quoted in GMT
            driver.find_elements(By.XPATH, "//a[@class='list-group-item single-line transcript']")[i].click()
            WebDriverWait(driver, 20).until(
                EC.frame_to_be_available_and_switch_to_it((By.XPATH, "//iframe[@id='embedded_doc']")))

            # text now contains the time in string form
            text: str = driver.find_elements(By.XPATH, "//p[contains(.,'GMT')]")[0].text[-14:-4]
            driver.execute_script("window.history.go(-1)")

            # determine whether it was before market open or after
            bmo: bool
            if text[-2:] == 'pm' and int(text[:2]) < 8:
                bmo = True
            elif text[-2:] == 'pm' and int(text[:2]) == 12:
                bmo = True
            elif text[-2:] == 'am':
                bmo = True
            else:
                bmo = False

            # save the time
            bmos.append(bmo)

            # rest for a moment to avoid being locked out
            tm.sleep(3)
            logging.debug(f'Resting after completing {right_labels[i].text} for {ticker}')

    # save results to current df and append to the df list
    df['Earnings Date']: pd.Series = dates
    df['BMO']: pd.Series = bmos
    df['Ticker']: pd.Series = ticker
    df_list.append(df)

    logging.info(f'Completed {ticker}')

    # take a more extended break to avoid being locked out
    tm.sleep(15)

# join all the individual dfs into a single one, and save results to parquet
final_df: pd.DataFrame = pd.concat(df_list)
final_df = final_df.astype({'Ticker': 'string', 'Earnings Date': 'datetime64', 'BMO': 'bool'})
final_df.to_parquet('EarningsCalendar.parquet')
