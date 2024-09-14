from flask import Flask, request, jsonify
import MetaTrader5 as mt5
import threading
import pandas as pd

login_id = int(5569074)
server = 'Deriv-Demo'
password = '6hPQkJsZ9@Wv3nr'
symbol = 'Volatility 50 Index'


def get_data(login_id, server, password, symbol):
    TIMEFRAME = mt5.TIMEFRAME_H4
    HISTORY_BARS = 10000

    if not mt5.initialize():
         {"error": f"initialize() failed, error code = {mt5.last_error()}"}

    authorized = mt5.login(login_id, password=password, server=server)
    
    if not authorized:
         {"error": f"Failed to connect to login_id {login_id}, error code: {mt5.last_error()}"}
    
    data = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, HISTORY_BARS)
    
    if data is None:
         {"error": f"Failed to retrieve data for symbol {symbol}, error code: {mt5.last_error()}"}
    
    # Mapping the structured array to a list of dictionaries
    data_list = []
    for record in data:
        data_list.append({
            "time": record.time,
            "open": record.open,
            "high": record.high,
            "low": record.low,
            "close": record.close,
            "tick_volume": record.tick_volume,
            "spread": record.spread,
            "real_volume": record.real_volume,
        })

    return {"data": data_list}



def api_get_data():
    login_id = request.json.get('login_id')
    server = request.json.get('server')
    password = request.json.get('password')
    symbol = request.json.get('symbol')
    
    if not all([login_id, server, password, symbol]):
         print({"status": "error", "message": "Missing required fields."}), 400

    data = get_data(login_id, server, password, symbol)
    
    if "error" in data:
         return(data), 500
    
print(api_get_data())