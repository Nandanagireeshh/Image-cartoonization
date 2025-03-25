import os
import io
import uuid
import sys
import yaml
import traceback

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

import cv2
from flask import Flask, render_template, make_response, flash
import flask
from PIL import Image
import numpy as np

if opts['colab-mode']:
    from flask_ngrok import run_with_ngrok  # type: ignore # to run the application on colab using ngrok

from cartoonize import WB_Cartoonize

app = Flask(__name__)
if opts['colab-mode']:
    run_with_ngrok(app)  # starts ngrok when the app is run

app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_images'
app.config['INTERMEDIATE_FOLDER'] = os.path.join(app.config['CARTOONIZED_FOLDER'], 'intermediate')
os.makedirs(app.config['INTERMEDIATE_FOLDER'], exist_ok=True)

app.config['OPTS'] = opts

## Init Cartoonizer and load its weights 
wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    """Convert bytes to numpy array

    Args:
        img_bytes (bytes): Image bytes read from flask.

    Returns:
        [numpy array]: Image numpy array
    """
    
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode=="RGBA":
        image = Image.new("RGB", pil_image.size, (255,255,255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    
    image = np.array(image)
    return image

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if flask.request.method == 'POST':
        try:
            if flask.request.files.get('image'):
                img = flask.request.files["image"].read()
                ## Read Image and convert to PIL (RGB) if RGBA convert appropriately
                image = convert_bytes_to_image(img)
                img_name = str(uuid.uuid4())
                
                # Get intermediate steps including final cartoon image
                steps = wb_cartoonizer.infer(image, return_steps=True)
                
                # Save each intermediate step to disk
                step_names = {}
                for key, img_out in steps.items():
                    save_path = os.path.join(app.config['INTERMEDIATE_FOLDER'], f"{img_name}_{key}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR))
                    step_names[key] = save_path

                # Use the final cartoon image for the main display
                cartoonized_img_name = os.path.join(app.config['CARTOONIZED_FOLDER'], img_name + ".jpg")
                cv2.imwrite(cartoonized_img_name, cv2.cvtColor(steps['cartoon'], cv2.COLOR_RGB2BGR))
                
                # if not opts["run_local"]:
                #     output_uri = upload_blob("cartoonized_images", cartoonized_img_name, img_name + ".jpg", content_type='image/jpg')
                #     os.system("rm " + cartoonized_img_name)
                #     cartoonized_img_name = generate_signed_url(output_uri)
                
                return render_template("index_cartoonized.html", 
                                       cartoonized_image=cartoonized_img_name,
                                       intermediate_images=step_names)
        
        except Exception:
            print(traceback.print_exc())
            flash("Our server hiccuped :/ Please upload another file! :)")
            return render_template("index_cartoonized.html")
    else:
        return render_template("index_cartoonized.html")

if __name__ == "__main__":
    if opts['colab-mode']:
        app.run()
    else:
        app.run(debug=False, host='127.0.0.1', port=int(os.environ.get('PORT', 8080)))
