import os 
from tensorflow.keras.models import load_model
import numpy as np

model = load_model(os.path.join(os.getcwd(), "mpg_model.h5"))
#model.summary()

x = np.zeros([1, 7])

x[0, 0] = 10 
x[0, 1] = 500
x[0, 2] = 100
x[0, 3] = 2500
x[0, 4] = 63
x[0, 5] = 78
x[0, 6] = 1

prediction = model.predict(x)
print(float(prediction[0]))





