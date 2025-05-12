from flask import Flask, render_template, request
from models.predictor import ProfessionPredictor
import pandas as pd

app = Flask(__name__)

predictor = ProfessionPredictor('data/professions.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            score = float(request.form.get('average_score'))
            subject = request.form.get('favorite_subject')

            input_df = pd.DataFrame({
                'average_score': [score],
                'favorite_subject': [subject]
            })

            result = predictor.model.predict(input_df)[0]

        except Exception as e:
            result = f"Ошибка: {e}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

