import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statistics import mode

app = Flask(__name__)

def preprocess_data(df, target_column):
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Perform any additional preprocessing steps here

    return X, y

def train_model(X_train, y_train):
    # Train a Random Forest classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def predict(df, target_column, X):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, df[target_column], test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Return the most common prediction value
    most_common_prediction = mode(predictions)

    return most_common_prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        print("test")
        # Get column names from the dataframe
        columns = df.columns.tolist()

        if 'target_column' in request.form:
            target_column = request.form['target_column']
        else:
            target_column = columns[0]  # Set default target column to the first column
        
        # Preprocess the data
        X, y = preprocess_data(df, target_column)

        # Make prediction
        prediction = predict(df, target_column, X)
        print("prediction :", prediction)

        return render_template('index.html', columns=columns, selected_target=target_column, prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
