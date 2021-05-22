from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import os
import numpy as np
import uuid

app = Flask(__name__)


DATA = {
 "cylinders" :{ "min": 3, "max": 8 } , 
 "displacement" :{ "min": 68.0, "max": 455.0 } , 
 "horsepower" :{ "min": 46.0, "max": 230.0 } ,
 "weight" :{ "min": 1613, "max": 5140 } ,
 "acceleration" :{ "min": 8.0, "max": 24.8 } ,
 "model year" :{ "min": 70, "max": 82 } ,
 "origin" :{ "min": 1, "max": 3 } 
}

os.chdir(r"E:\Gülfem\ML_Coding")
model = load_model(os.path.join(os.getcwd(), "mpg_model.h5"))

@app.route('/api', methods=['POST'])

def x_prediction():
    content = request.json
    errors = []

    for name in content:
        if name in DATA:
            data_min = DATA[name]['min']
            data_max = DATA[name]['max']
            value = DATA[name]
            if name > data_max or name < data_min:
                errors.append(f"Out of bounds: {name}, has value of: {value}, but it should be between {data_min} and {data_max}")
        else:
            errors.append(f"Unexpected field: {name}.")

        for name in DATA:
            if name not in content:
                errors.append(f"missing_value: {name}")

        if len(errors) < 1:
            x = np.zeros([1,7])
            x[0, 0] = content['cylinders']
            x[0, 1] = content['displacement']
            x[0, 2] = content['horsepower']
            x[0, 3] = content['weight']
            x[0, 4] = content['acceleration']
            x[0, 5] = content['year']
            x[0, 6] = content['origin']

            prediction = model.predict(x)
            mpg = float(prediction[0])
            response = { "id": str(uuid.uuid4()), "mpg":mpg, "errors":errors }
        else:
            response = { "id": str(uuid.uuid4(), errors = errors)}
        
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)







