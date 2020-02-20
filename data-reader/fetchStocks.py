from pandas_datareader import data
import pandas as pd

if __name__ == '__main__':

    # Initialization
    tickers = ['^GSPC', 'XOM', 'AAPL', 'INTC', 'GS', 'GE', 'AXP', 'WMT', 'PG']
    start_date = '2000-01-01'
    end_date = '2019-12-31'

    # Fetch Data from yahoo api
    panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)

    """
    Note: Yahoo! API was deprecated and multi-index form of data is returned
    in the 0.7.0 version of pandas-datareader. We use the following method to 
    return the ideal form of panel data as follows. The method is not optimal
    but intuitional.
    """

    panel_data2 = pd.DataFrame()
    for ticker in tickers:
        cols_ticker = []
        for col in panel_data:
            if col[1] == ticker:
                cols_ticker.append(col)
        _df = panel_data[cols_ticker]
        _df.columns = [col[0] for col in cols_ticker]
        _df['Symbol'] = ticker
        panel_data2 = panel_data2.append(_df)
    panel_data2['Date'] = panel_data2.index
    panel_data2 = panel_data2[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    # Show the head of panel data
    print(panel_data2.head(10))

    # Store the sample data
    panel_data2.to_csv('../data/raw/sample-stocks.csv', index=False)
