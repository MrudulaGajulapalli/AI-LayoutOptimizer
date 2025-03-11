from flask import Flask, render_template, request, jsonify
from stable_baselines3 import PPO
from room_env import FurnitureArrangementEnv

app = Flask(__name__)

# Load trained model
model = PPO.load("furniture_model.zip")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    width = int(request.args.get("width", 10))
    height = int(request.args.get("height", 10))

    env = FurnitureArrangementEnv(room_size=(width, height))
    obs = env.reset()
    action, _ = model.predict(obs)
    furniture_positions = env.furniture_positions.tolist()

    return render_template('result.html', room_size=[width, height], furniture_positions=furniture_positions)

if __name__ == '__main__':
    app.run(debug=True)
