from flask import Flask, request, render_template, send_file, safe_join
import genscript  as gs
import torch
from zipfile import ZipFile
import os
from os.path import basename


app = Flask(__name__)

@app.route('/')
def root_page():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    num = request.form['num']
    num = int(num.upper())
    istate = request.form['state_input']
    gs.make_plate_images(num, istate)



    return get_zip()


def get_zip():
    # create a ZipFile object
    with ZipFile('images.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk('generated_images'):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, basename(filePath))

    try:
        return send_file('./images.zip', as_attachment=True)
    except FileNotFoundError:
        abort(404)

