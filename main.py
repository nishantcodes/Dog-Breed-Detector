from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import numpy as np
import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite


model = tflite.Interpreter("static/model.tflite")
model.allocate_tensors()

input_details = model.get_input_details()
output_details = model.get_output_details()

class_mapping = {0: 'Afghan',
 1: 'African Wild Dog',
 2: 'Airedale',
 3: 'American Hairless',
 4: 'American Spaniel',
 5: 'Basenji',
 6: 'Basset',
 7: 'Beagle',
 8: 'Bearded Collie',
 9: 'Bermaise',
 10: 'Bichon Frise',
 11: 'Blenheim',
 12: 'Bloodhound',
 13: 'Bluetick',
 14: 'Border Collie',
 15: 'Borzoi',
 16: 'Boston Terrier',
 17: 'Boxer',
 18: 'Bull Mastiff',
 19: 'Bull Terrier',
 20: 'Bulldog',
 21: 'Cairn',
 22: 'Chihuahua',
 23: 'Chinese Crested',
 24: 'Chow',
 25: 'Clumber',
 26: 'Cockapoo',
 27: 'Cocker',
 28: 'Collie',
 29: 'Corgi',
 30: 'Coyote',
 31: 'Dalmation',
 32: 'Dhole',
 33: 'Dingo',
 34: 'Doberman',
 35: 'Elk Hound',
 36: 'French Bulldog',
 37: 'German Sheperd',
 38: 'Golden Retriever',
 39: 'Great Dane',
 40: 'Great Perenees',
 41: 'Greyhound',
 42: 'Groenendael',
 43: 'Irish Spaniel',
 44: 'Irish Wolfhound',
 45: 'Japanese Spaniel',
 46: 'Komondor',
 47: 'Labradoodle',
 48: 'Labrador',
 49: 'Lhasa',
 50: 'Malinois',
 51: 'Maltese',
 52: 'Mex Hairless',
 53: 'Newfoundland',
 54: 'Pekinese',
 55: 'Pit Bull',
 56: 'Pomeranian',
 57: 'Poodle',
 58: 'Pug',
 59: 'Rhodesian',
 60: 'Rottweiler',
 61: 'Saint Bernard',
 62: 'Schnauzer',
 63: 'Scotch Terrier',
 64: 'Shar_Pei',
 65: 'Shiba Inu',
 66: 'Shih-Tzu',
 67: 'Siberian Husky',
 68: 'Vizsla',
 69: 'Yorkie'}


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



def resize(image):
    return cv2.resize(image, (224,224))

def model_predict(images_arr):
    predictions = [0]* len(images_arr)
    
    for i, val in enumerate(predictions):
        model.set_tensor(input_details[0]['index'], images_arr[i].reshape(1, 224, 224, 3))
        model.invoke()
        predictions[i] = model.get_tensor(output_details[0]['index']).reshape((70,))
        
    prediction_probs = np.array(predictions)
    argmaxs = np.argmax(prediction_probs, axis=1)
    
    return argmaxs


@app.post('/uploadfiles/', response_class=HTMLResponse)
async def create_upload_files(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        f = await file.read()
        images.append(f)

    images = [np.frombuffer(img, np.uint8) for img in images]
    images = [cv2.imdecode(img, cv2.IMREAD_COLOR) for img in images]
    images_resized = [resize(img) for img in images]
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_resized]

    names = [file.filename for file in files]

    for image, name in zip(images_rgb, names):
        pillow_image = Image.fromarray(image)
        pillow_image.save('static/'+ name)
    
    image_paths = ['static/' + name for name in names]

    images_arr = np.array(images_rgb, dtype = np.float32)
    
    class_indexes = model_predict(images_arr)

    class_predictions = [class_mapping[x] for x in class_indexes]

    column_labels = ["Image", "Prediction"]

    table_html = get_html_table(image_paths, class_predictions, column_labels)

    content = head_html + '<h4 class="center header col s12 #bdbdbd red-text lighten-1"> Here Are Our Predictions.</h4>'+ str(table_html) + """
    <br><form method = "post" action="/">
    <div class = "row center">
    <button type="submit" class="btn waves-effect waves-light green">Home</button>
    </div>
    </form>
    """

    return content

@app.post("/", response_class=HTMLResponse)
@app.get('/', response_class=HTMLResponse)
async def main():
    content = head_html+"""
    <div class="row center">
        <h5 class="header col s12 #bdbdbd grey-text lighten-1">How often do you see a doggo and wanted to know it's breed. If this is the case, you are just 1 click away. These are not absolutely correct results, Our testing has shown '95%' accuracy.  
             <br>
        </h5>
    </div>
    <h4 class="center header col s12 #bdbdbd red-text lighten-1"> Please upload your Dog Pictures below.
    <h6 class="center header col s12 #bdbdbd grey-text lighten-1"> We'll try to predict your doggo's breed like these:
    <br><br>
    """

    original_paths = ['Beagle.jpg', 'Boxer.jpg', 'Bulldog.jpg', 'Corgi.jpg', 'Doberman.jpg', 'Labrador.jpg','Pitbull.jpg','Poodle.jpg','SiberianHusky.jpg']
    
    full_original_paths = ['static/original/'+ x for x in original_paths]

    display_names = ['Beagle', 'Boxer', 'Bulldog', 'Corgi', 'Doberman', 'Labrador','Pitbull','Poodle','Siberian Husky']

    column_labels = []

    content = content + get_html_table(full_original_paths, display_names, column_labels)

    content = content + """
    <br/>
    <br/>
    <form action = "/uploadfiles/" enctype = "multipart/form-data" method = "post">
    <input name = "files" type = "file" multiple>
    <button type="submit" class="btn waves-effect waves-light green">Submit</button>
    </form>
    </body>

    """
    return content


head_html = """
<head>
    <met name = "viewport" content="width=device-width, initial-scale=1"/>\
    <title>🐶 Dog Breed Guesser</title>   
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script> 
</head>
<body>
    <nav class="#304ffe brown accent-4" role="navigation">
        <div class="nav-wrapper container"><a id="logo-container" href="https://www.linkedin.com/in/nishant-dhingra-82515918a/" class="brand-logo">Nishant👨‍💻</a>
        </div>
    </nav>
    <div class="section no-pad-bot" id="index-banner">
        <div class="container">
            <br><br>
            <h1 class="header center brown-text">🐶 Dog Breed Guesser 📸</h1>  
"""

def get_html_table(image_paths, names, column_labels):
    s = '<table align="center">'
    if column_labels:
        s+= '<tr><th><div class="center"><h5 class="header col s12 #bdbdbd grey-text lighten-1">' + column_labels[0] + '</h5></div></th><th><div class="center"><h5 class="header col s12 #bdbdbd grey-text lighten-1">' + column_labels[1] + '</h5></div></th></tr>'
    
    for name, image_path in zip(names, image_paths):
        s += '<tr><td><div class="center"><img height="120" src="/' + image_path + '"></div></td>'
        s += '<td><div class="center"><h6>' + name + '</h6></div></td></tr>'
    s+='</table>'

    return s


