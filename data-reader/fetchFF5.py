from pandas_datareader import data

if __name__ == '__main__':

    # Initialization
    ticker = 'F-F_Research_Data_5_Factors_2x3_daily'
    start_date = '2000-01-01'
    end_date = '2019-12-31'

    # Fetch Data from yahoo api
    res = data.DataReader(ticker, 'famafrench', start_date, end_date)
    """
    Additional Information:
    
    F-F Research Data 5 Factors 2x3 daily
    -------------------------------------
    
    This file was created by CMPT_ME_BEME_OP_INV_RETS_DAILY using the 201902 CRSP database. The 1-month TBill return is from Ibbotson and Associates, Inc.
    
      0 : (3775 rows x 6 cols)
    
    """

    # Show the head of panel data
    panel_data = res[0]
    print(panel_data.head(10))

    # Store the sample data
    panel_data.to_csv('../data/raw/sample-ff5.csv')
