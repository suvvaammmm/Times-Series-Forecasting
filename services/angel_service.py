from SmartApi import SmartConnect
import pyotp
import pandas as pd
from datetime import datetime, timedelta
from config import API_KEY, CLIENT_ID, PASSWORD, TOTP_SECRET


def get_angel_data(symbol_token):

    obj = SmartConnect(api_key=API_KEY)

    totp = pyotp.TOTP(TOTP_SECRET).now()
    obj.generateSession(CLIENT_ID, PASSWORD, totp)

    to_date = datetime.now()
    from_date = to_date - timedelta(days=365)

    params = {
        "exchange": "NSE",
        "symboltoken": symbol_token,
        "interval": "ONE_DAY",
        "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
        "todate": to_date.strftime("%Y-%m-%d %H:%M")
    }

    response = obj.getCandleData(params)

    df = pd.DataFrame(response['data'],
                      columns=['Datetime','Open','High','Low','Close','Volume'])

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    return df['Close']
