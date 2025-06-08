import os
import sys
import api
from api import app
#from api import house_price_api
from api import house_price


if __name__ == '__main__':
    app.debug = False
    port = int(os.environ.get('PORT', 5000))
    app.run()
