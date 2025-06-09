from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    
    # Konfigurasi upload
    UPLOAD_FOLDER = 'app/static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Pastikan folder upload ada
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs('app/static/images', exist_ok=True)
    
    # Load model dengan error handling
    try:
        model = load_model('app/model/model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess_image(image_path):
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))  
            img = np.array(img) / 255.0  
            img = np.expand_dims(img, axis=0)  
            return img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/scantype', methods=['GET', 'POST'])
    def skinType():
        if request.method == 'POST':
            
            if model is None:
                return render_template('skintype.html', 
                                     error="Model tidak dapat dimuat. Silakan coba lagi nanti.")
            
            file = request.files.get('image')
            
            if not file or file.filename == '':
                return render_template('skintype.html', 
                                     error="Tidak ada file yang dipilih")
            
            if file and allowed_file(file.filename):
                try:
                    # Generate unique filename untuk menghindari konflik
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4().hex}_{filename}"
                    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
                    
                    
                    file.save(filepath)
                    
                    
                    img = preprocess_image(filepath)
                    if img is None:
                        return render_template('skintype.html', 
                                             error="Gagal memproses gambar")
                    
                    # Prediksi
                    preds = model.predict(img)
                    class_idx = np.argmax(preds)
                    confidence = preds[0][class_idx]
                    
                    classes = ['Normal', 'Oily', 'Dry']
                    predicted_class = classes[class_idx]
                    
                    return render_template('skintype.html',
                                         filename=unique_filename,
                                         predicted_class=predicted_class,
                                         confidence=round(confidence * 100, 2))
                                         
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    return render_template('skintype.html', 
                                         error="Terjadi kesalahan saat menganalisis gambar")
            else:
                return render_template('skintype.html', 
                                     error="Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau GIF")
        else:
            # GET request - render form kosong
            return render_template('skintype.html')

    @app.route('/result')
    def details():
        return render_template('single-skintype.html')

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)