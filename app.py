import numpy as np
import scipy as sp
from scipy import misc

import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3,preprocess_input
import os


from flask import Flask,render_template,request,url_for,flash, redirect
from flask_bootstrap import Bootstrap  
from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField,BooleanField

#from werkzeug import secure_filename
import time

img_dim = 512
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 4
f_size = img_dim // 2**(num_layers + 1)
batch_size = 32

if not os.path.exists('static'):
    os.mkdir('static')


def sample(path,generate_model,n=9,z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = generate_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)
    #return figure

def generator(num_samples):
    
    model_dir="./g_train_model.best.weights"
    image_dir="./static/demo.jpg"
    img_dim = 512
    z_dim = 128
    num_layers = int(np.log2(img_dim)) - 3
    max_num_channels = img_dim * 4
    f_size = img_dim // 2**(num_layers + 1)


    # In[9]:


    # generator
    z_in = Input(shape=(z_dim, ))
    z = z_in

    z = Dense(f_size**2 * max_num_channels)(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = Reshape((f_size, f_size, max_num_channels))(z)

    for i in range(num_layers):
        num_channels = max_num_channels // 2**(i + 1)
        z = Conv2DTranspose(num_channels,
                            (5, 5),
                            strides=(2, 2),
                            padding='same')(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)

    z = Conv2DTranspose(3,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = Activation('tanh')(z)

    g_model = Model(z_in, z)


    # discriminator
    x_in = Input(shape=(img_dim, img_dim, 3))
    x = x_in

    for i in range(num_layers + 1):
        num_channels = max_num_channels // 2**(num_layers - i)
        x = Conv2D(num_channels,
                   (5, 5),
                   strides=(2, 2),
                   use_bias=False,
                   padding='same')(x)
        if i > 0:
            x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    x = Dense(1, use_bias=False)(x)

    d_model = Model(x_in, x)


    x_in = Input(shape=(img_dim, img_dim, 3))
    z_in = Input(shape=(z_dim, ))
    g_model.trainable = False

    x_real = x_in
    x_fake = g_model(z_in)

    x_real_score = d_model(x_real)
    x_fake_score = d_model(x_fake)

    x_in = Input(shape=(img_dim, img_dim, 3))
    z_in = Input(shape=(z_dim, ))
    g_model.trainable = False

    x_real = x_in
    x_fake = g_model(z_in)

    x_real_score = d_model(x_real)
    x_fake_score = d_model(x_fake)

    g_train_model = Model([x_in, z_in],
                          [x_real_score, x_fake_score])

    # load pre-trained model
    g_train_model.load_weights(model_dir)
    
    sample(path=image_dir,generate_model=g_model,n=num_samples)

# create form function
class ReusableForm(Form):
    """User entry form for entering specifics numbers for generation"""
    num_samples=IntegerField("Enter number of samples to generate GAN demo picture:",
                     default=3,validators=[validators.InputRequired(),
                                    validators.NumberRange(min=1,max=16,
                                    message="Number of samples must be bwtween 1 and 16")])
    
    update=BooleanField("update",validators=[validators.DataRequired()])
    # Submit button
    submit = SubmitField("Enter")
    
  


app=Flask(__name__)
# Home page
@app.route("/",methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
def index():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)
    if request.method=="POST":
       num_samples=int(request.form["num_samples"])
       generator(num_samples)
    if form.validate():
        update=form.update.data
        if update:
          flash('Update !')  
       
    # Send template information to index.html
    return render_template('index.html', form=form)                                         

#@app.route("/",methods=["GET","POST"])
@app.route("/gan",methods=["GET","POST"])
def gan():
      return render_template("gan.html")
                        

    
if __name__=="__main__":
    app.run(host="10.100.110.101",debug=True)
