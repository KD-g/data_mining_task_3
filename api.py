from flask import Flask, request, render_template
import pandas as pd
import joblib
from assets_data_prep import prepare_data
import numpy as np

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load('trained_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Extract form data
            num_of_images = float(request.form.get('num_of_images', 0))
            is_renovated = int(request.form.get('is_renovated'))
            furnished = int(request.form.get('furnished'))
            has_balcony = int(request.form.get('has_balcony'))
            has_safe_room = int(request.form.get('has_safe_room'))
            has_bars = int(request.form.get('has_bars'))
            handicap = int(request.form.get('handicap'))
            ac = int(request.form.get('ac'))
            elevator = int(request.form.get('elevator'))
            has_storage = int(request.form.get('has_storage'))
            parking = int(request.form.get('parking'))
            description = str(request.form.get('description'))
            total_floors = float(request.form.get('total_floors'))
            building_tax = float(request.form.get('building_tax'))
            monthly_arnona = float(request.form.get('monthly_arnona'))
            num_of_payments = float(request.form.get('num_of_payments'))
            days_to_enter = float(request.form.get('days_to_enter'))
            garden_area = float(request.form.get('garden_area'))
            area = float(request.form.get('area'))
            floor = int(request.form.get('floor'))
            num_rooms = float(request.form.get('num_rooms'))
            address = str(request.form.get('address'))
            neighborhood = str(request.form.get('neighborhood'))
            property_type = str(request.form.get('property_type'))
            
            # Create a single-row DataFrame for the model
            input_df = pd.DataFrame([{
                'property_type': property_type,
                'neighborhood': neighborhood,
                'address': address,
                'room_num': num_rooms,
                'floor': floor,
                'area': area, 
                'garden_area': garden_area,
                'days_to_enter': days_to_enter,
                'num_of_payments': num_of_payments,
                'monthly_arnona': monthly_arnona,
                'building_tax': building_tax,
                'total_floors': total_floors,
                'description': description, 
                'has_parking': parking,
                'has_storage': has_storage,
                'elevator': elevator, 
                'ac': ac,
                'handicap': handicap, 
                'has_bars': has_bars, 
                'has_safe_room': has_safe_room,
                'has_balcony': has_balcony,
                'is_furnished': furnished,
                'is_renovated': is_renovated,
                'num_of_images': num_of_images,
                'distance_from_center': np.nan
            }])

            # Preprocess and predict
            processed_df = prepare_data(input_df)

            pred_value = model.predict(processed_df)[0]

            prediction = round(pred_value, 2)
        except Exception as e:
            # Capture and display any error
            error_message = str(e)
    
    # Render the form template with prediction or error message (if any)
    return render_template('index.html', prediction=prediction, error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
