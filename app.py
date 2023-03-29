from flask import Flask, request, jsonify, send_from_directory
import json
import os
import shutil
import sys
import predictfood
import predictclasses
import base64

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/image-receiver', methods=['POST'])
def image_receiver():
    print('test')
    # Create a temporary directory to save the files
    TEMP_DIR = os.path.abspath('temp-download')
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

    print("Saving local files:")
    based = bytes(request.json['baseString'][23:], 'utf-8')
    #print(based)
    with open("temp-download/temp.jpg", "wb") as fh:
        fh.write(base64.decodebytes(based))
        #files[field_name.replace('-', '_')] = os.path.join('temp-download', f.filename.replace('-', '_'))
        #print('Saved ', field_name, files[field_name.replace('-', '_')])

        # json_data[]
    #print(files)
    print('Saved all files locally to the temp-download directory')
    #call model function
    #return output
    filepath = './temp-download/temp.jpg'
    food = predictfood.predictfood("./temp-download/temp.jpg")
    classes = predictclasses.predictclasses(filepath)

    return [food, classes]

if __name__ == '__main__':
    app.run(host='0.0.0.0')