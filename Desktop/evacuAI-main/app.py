from flask import Flask
import threading
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

from api.routes import bp
app.register_blueprint(bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
