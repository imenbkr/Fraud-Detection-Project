from flask import Flask

# Create a Flask app instance
app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def hello_world():
    return 'Hello, Flask!'

# Run the app when the script is executed
if __name__ == '__main__':
    app.run()
