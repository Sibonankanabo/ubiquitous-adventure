from flask import Flask, request, jsonify
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    login_id = int(data.get('login_id'))
    password = data.get('password')
    server = data.get('server')

    if mt5.initialize():
        authorized = mt5.login(login_id, password=password, server=server)
        if authorized:
            return jsonify({"status": "success", "message": "Logged in successfully."})
        else:
            return jsonify({"status": "error", "message": "Wrong credentials."}), 401
    else:
        return jsonify({"status": "error", "message": "Failed to initialize MetaTrader5."}), 500

@app.route('/symbols', methods=['GET'])
def get_symbols():
    if not mt5.initialize():
        return jsonify({"status": "error", "message": "Failed to initialize MetaTrader5."}), 500
    
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols]
    return jsonify(symbol_names)

def process_data(df):
    """
    Function to process and calculate technical indicators, scale, and reshape data.
    Returns a dictionary containing the transformed features (X), target (y), and the original unscaled data.
    """
    # Calculate indicators
    df['RSI'] = ta.rsi(df['close'], length=9)
    df['EMAF'] = ta.ema(df['close'], length=10)
    df['EMAM'] = ta.ema(df['close'], length=26)
    df['EMAS'] = ta.ema(df['close'], length=50)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=10)

    df['Adj Close'] = df['close']
    df['TargetNextClose'] = df['close'].shift(-1)
    df['target'] = df['TargetNextClose'] - df['close']
    # Drop columns we don't need and that can't be scaled
    df.drop(['tick_volume', 'spread', 'real_volume', 'close', 'time'], axis=1, inplace=True)
    df.dropna(inplace=True)
   
    # Keep the unaltered version of the dataset for X_train_original
    X_train_original = df.copy()  # Make a copy before modifying df further
    X_train_original = X_train_original.drop('TargetNextClose', axis=1)
    # Fit and transform the data using MinMaxScaler
    scaler = MinMaxScaler()

    # Drop 'TargetNextClose' only for the feature scaling
    scaled_data = scaler.fit_transform(df.drop('target', axis=1))  # Dropping the target column here
    df_scaled = pd.DataFrame(scaled_data, columns=df.drop('target', axis=1).columns)

    # Define features (X) and target (y)
    X = df_scaled  # X contains all scaled features except the target
    y = df['target']  # Target remains unchanged

    # Convert to numpy arrays
    X = np.array(X)  # X shape: (n_samples, n_features)
    y = np.array(y)  # y shape: (n_samples,)

    return X, y, scaler, df_scaled, X_train_original


def retrieve_mt5_data(login_id, password, server, symbol, timeframe=mt5.TIMEFRAME_H4, history_bars=10000):
    """
    Helper function to retrieve and preprocess data from MetaTrader 5.
    Returns a processed DataFrame.
    """
    if not mt5.initialize():
        return {"error": f"initialize() failed, error code = {mt5.last_error()}"}

    authorized = mt5.login(login_id, password=password, server=server)
    if not authorized:
        return {"error": f"Failed to connect to login_id {login_id}, error code: {mt5.last_error()}"}
    
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, history_bars)
    if data is None or len(data) == 0:
        return {"error": "No data retrieved."}
    
    # Convert data to DataFrame and process
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['tick_volume'] = df['tick_volume'].astype(float)
    df.sort_values('time', inplace=True)

    return df

@app.route('/get_data', methods=['POST'])
def api_get_data():
    login_id = request.json.get('login_id')
    server = request.json.get('server')
    password = request.json.get('password')
    symbol = request.json.get('symbol')
    
    if not all([login_id, server, password, symbol]):
        return jsonify({"status": "error", "message": "Missing required fields."}), 400

    # Retrieve data
    df = retrieve_mt5_data(login_id, password, server, symbol)
    if "error" in df:
        return jsonify(df), 500
    
    # Process data
    X, y, scaler, df_scaled,X_train_original = process_data(df)
    
    # Split the data into training and testing sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(X_train.shape)
    # Convert numpy arrays to lists to make them JSON serializable
    response_data = {
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist(),
        'X_train_original': X_train_original.values.tolist(),

    }

    return jsonify(response_data)

@app.route('/get_trading_data', methods=['POST'])
def get_trading_data():
    
    data = request.json
    login_id = data.get('login_id')
    server = data.get('server')
    password = data.get('password')
    symbol = data.get('symbol')
    
    
    if not all([login_id, server, password, symbol]):
        return jsonify({"status": "error", "message": "Missing required fields."}), 400

    # Retrieve data
    df = retrieve_mt5_data(login_id, password, server, symbol)
    if "error" in df:
        return jsonify(df), 500

    # Process data
    X_new_data, _, scaler, df_scaled,X_train_original = process_data(df)
    # print(X_new_data.shape)
    # Get current price (last known EMAF close)
    current_price = df['EMAF'].iloc[-1]
    
    # Get the latest ATR value
    atr = df['ATR'].iloc[-1]

    # Prepare response data
    
    response_data = {
        'X_new_data': X_new_data.tolist(),  # X_new_data is now 3D for LSTM models
        'current_price': current_price,
        'atr': atr,
        'scaled_features': df_scaled.iloc[-1].tolist(),
        'X_train_original': X_train_original.values.tolist(),

    }

    return jsonify(response_data)

@app.route('/close_position', methods=['POST'])
def close_position():
    data = request.json
    login_id = data.get('login_id')
    password = data.get('password')
    server = data.get('server')
    symbol = data.get('symbol')
    order_type = data.get('order_type')
    

    if not all([login_id, password, server, symbol]):
        return jsonify({"status": "error", "message": "Missing required fields."}), 200
    
    
    # Fetch open positions
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return jsonify({"status": "error", "message": "No open positions found."}), 200
    
    
    # Close all positions for the symbol
    results = []
    for position in positions:
        order_type = mt5.ORDER_TYPE_BUY if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "magic": position.magic,
            "comment": "LSTM Close Position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(close_request)
        results.append(result._asdict())

    return jsonify({"status": "success", "message": "Positions closed successfully.", "results": results})

@app.route('/place_order', methods=['POST'])
def place_order():
    data = request.json
    login_id = data.get('login_id')
    password = data.get('password')
    server = data.get('server')
    symbol = data.get('symbol')
    volume = data.get('volume')
    order_type = data.get('order_type')
    price = data.get('price')
    sl = data.get('sl')
    
    
    
    if not all([login_id, password, server, symbol, volume, order_type, price, sl]):
        
        return jsonify({"status": "error", "message": "Missing required fields."}), 404

    # Ensure MetaTrader5 is initialized and logged in

    # Determine order type
    if order_type.lower() == 'buy':
        order_type_mt5 = mt5.ORDER_TYPE_BUY
    elif order_type.lower() == 'sell':
        order_type_mt5 = mt5.ORDER_TYPE_SELL
    else:
        print(order_type)
        return jsonify({"status": "error", "message": "Invalid order type."}), 400

    # Prepare order request
    order_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_mt5,
        "price": price,
        "sl": sl,
        "magic": 0,  # Set to your magic number if needed
        "comment": "LSTM Trade Order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    # Place the order
    result = mt5.order_send(order_request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return jsonify({"status": "error", "message": f"Failed to place order: {result.retcode}"}), 500

    return jsonify({"status": "success", "message": "Order placed successfully.", "result": result._asdict()})

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
