# views.py
from django.shortcuts import render
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model('path/to/model.h5')

def predict_view(request):
    if request.method == 'POST' and request.FILES['image']:
        image = Image.open(request.FILES['image']).resize((224, 224))
        image = np.array(image) / 255.0
        image = image.reshape(1, 224, 224, 3)
        prediction = model.predict(image)
        label = np.argmax(prediction)
        return render(request, 'result.html', {'label': label})
    return render(request, 'upload.html')
