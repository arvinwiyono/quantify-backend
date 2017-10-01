from flask import Flask
from flask import request
from sklearn.externals import joblib
from flask import jsonify
import sys
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

df = pd.read_csv('./data/grouped-suburb-with-locations.csv')
buy_scaler = joblib.load('./models/price/buy_scaler.pkl')
buy_predictor = joblib.load('./models/price/random_forest.pkl')

rent_scaler = joblib.load('./models/rent/rent_scaler.pkl')
rent_predictor = joblib.load('./models/rent/gradient_boosting.pkl')

@app.route('/')
def hello_world():
    return 'Hello, Quantify!'

@app.route('/api', methods=['POST'])
def get_suburb_data():
    data = request.get_json()
    print(data, file=sys.stderr)
    year_start = int(data['year_start'])
    year_end = int(data['year_end'])
    num_bedrooms = int(data['num_bedrooms'])
    prop_type = data['property_type'].upper()
    suburb = data['suburb'].upper()

    geo_data = df[df.suburb == suburb]
    geo_data = geo_data[geo_data.property_type == prop_type]
    geo_data = geo_data[geo_data.num_bedrooms == num_bedrooms].iloc[0]

    lat = geo_data.lat
    long = geo_data.lon

    columns = ['lat', 'long', 'num_bedrooms', 'year', 'property_type']
    dataset = pd.DataFrame(columns=columns)
    for year in range(year_start, year_end+1):
        dataset = dataset.append(pd.Series([lat, long, num_bedrooms, year, prop_type], index=columns), ignore_index=True)
    # tricking the dummies
    dataset = dataset.append(pd.Series([lat, long, 9999, num_bedrooms, 'HOUSE'], index=columns), ignore_index=True)
    dataset = dataset.append(pd.Series([lat, long, 9999, num_bedrooms, 'APARTMENT'], index=columns), ignore_index=True)

    dataset = pd.concat([dataset, pd.get_dummies(dataset.property_type, prefix='prop_type')], axis=1)
    dataset = dataset.drop(['property_type', 'prop_type_APARTMENT'], axis=1)
    print(dataset, file=sys.stderr)
    print('**********', file=sys.stderr)
    buy_x = buy_scaler.transform(dataset.iloc[:-2])
    print(buy_x, file=sys.stderr)
    print('**********', file=sys.stderr)
    rent_x = rent_scaler.transform(dataset.iloc[:-2])

    predicted_price = buy_predictor.predict(buy_x)
    for i, p in enumerate(predicted_price):
        predicted_price[i] = p + 0.021 * i * p
    predicted_rent = rent_predictor.predict(rent_x)
    for i, p in enumerate(predicted_rent):
        predicted_rent[i] = p + 0.019 * i * p
    print(lat, long, file=sys.stderr)
    print(dataset, file=sys.stderr)

    print(buy_x, file=sys.stderr)
    print(rent_x, file=sys.stderr)

    print(predicted_price, file=sys.stderr)
    print(predicted_rent, file=sys.stderr)

    output = {
        'suburb': suburb,
        'lat': lat,
        'long': long,
        'property_type': prop_type,
        'num_bedrooms': num_bedrooms
    }

    out_cols = [
        'n_transport_1km',
        'n_school_1km',
        'n_food_1km',
        'n_shop_1km',
        'n_hospital_1km',
        'n_landmark_1km',
        'n_transport_3km',
        'n_school_3km',
        'n_food_3km',
        'n_shop_3km',
        'n_hospital_3km',
        'n_landmark_3km',
    ]
    for col in out_cols:
        output[col] = geo_data[col]
    print(output, file=sys.stderr)

    for i, year in enumerate(range(year_start, year_end+1)):
        print(i, year, file=sys.stderr)
        output[str(year)] = {}
        output[str(year)]['price'] = predicted_price[i]
        output[str(year)]['rent'] = predicted_rent[i]

    print(output, file=sys.stderr)
    return jsonify(output)
