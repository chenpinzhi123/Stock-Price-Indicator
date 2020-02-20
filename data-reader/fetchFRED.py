from pandas_datareader import data

if __name__ == '__main__':


    # Initialization
    tickers = [# 'A191RL1Q225SBEA', # real GDP growth: Percent Change from Preceding Period, Seasonally Adjusted Annual Rate
               'USD12MD156N', # 12-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
               'DTWEXM', # Trade Weighted U.S. Dollar Index: Major Currencies, Goods; Index Mar 1973=100, Not Seasonally Adjusted
               # 'M2', # M2 Monetary Stock; Billions of Dollars, Seasonally Adjusted
               'DAAA', # Moody's Seasoned Aaa Corporate Bond Yield, Percent, Not Seasonally Adjusted
               'VIXCLS', # CBOE Volatility Index: VIX; Index, Not Seasonally Adjusted
               # 'RECPROUSM156N', # Smoothed U.S. Recession Probabilities, Percent, Not Seasonally Adjusted
              ]

    start_date = '2000-01-01'
    end_date = '2019-12-31'

    # Fetch Data from yahoo api
    panel_data = data.DataReader(tickers, 'fred', start_date, end_date)

    # Show the head of panel data
    print(panel_data.head(10))

    # Store the sample data
    panel_data.to_csv('../data/raw/sample-fred.csv')
