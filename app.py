from flask import Flask, render_template, request
from models.use_model import lr_predict

app = Flask(__name__)


@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Get user input values from form,
    and return prediction back to interface.
    """
    if request.method == 'POST':
        # Collecting user input values from the form.
        beer_style = request.form['beer_style']
        sku = request.form['sku']
        fermentation_time = float(request.form['fermentation'])
        temperature = float(request.form['temperature'])
        ph_level = float(request.form['ph'])
        gravity = float(request.form['gravity'])
        alcohol_content = float(request.form['alcohol_content'])
        bitterness = float(request.form['bitterness'])
        color = float(request.form['color'])
        ingredient_ratio = "1:" + request.form['ingredient_ratio']
        volume_produced = float(request.form['volume_produced'])

        # Make prediction for quality score and total sales.
        prediction = lr_predict(beer_style, sku, fermentation_time, temperature, ph_level, gravity,
                                alcohol_content, bitterness, color, ingredient_ratio, volume_produced)
        return render_template('index.html', prediction=prediction)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
