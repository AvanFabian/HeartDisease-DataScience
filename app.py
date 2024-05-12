from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)

        # Get column names from the dataframe
        columns = df.columns.tolist()

        if 'target_column' in request.form:
            target_column = request.form['target_column']
        else:
            target_column = columns[0]  # Set default target column to the first column

        return render_template('index.html', columns=columns, selected_target=target_column)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
