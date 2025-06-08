from flask import Flask
app = Flask(__name__)
# app.config.from_object("config.DevelopmentConfig")

#from api import house_price_api
from api import house_price


