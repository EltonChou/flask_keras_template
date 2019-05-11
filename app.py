from flask import Flask, jsonify, request
from flask_restful import Api, Resource

from controllers.Model import Model


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        data = {
            "msg": "Hello, World."
        }
        return jsonify(data)


api.add_resource(HelloWorld, '/')
api.add_resource(Model, '/predict')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
