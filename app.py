import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from statistics import mode
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

app = Flask(__name__)

def preprocess_data(df, target_column):
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Convert categorical features to numerical values
    categorical_columns = X.select_dtypes(include=['object']).columns

    for column in categorical_columns:
        # Apply one-hot encoding to nominal categorical features
        encoder = OneHotEncoder()
        encoded_columns = encoder.fit_transform(X[[column]])
        encoded_df = pd.DataFrame(encoded_columns.toarray(), columns=[f"{column}_{category}" for category in encoder.categories_[0]])
        X = pd.concat([X.drop(columns=[column]), encoded_df], axis=1)

    # Convert target column if it is categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X, y


def train_model(X_train, y_train):
    # Train a Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,        # Number of trees in the forest
        max_depth=None,          # Maximum depth of the trees
        min_samples_split=2,     # Minimum number of samples required to split a node
        min_samples_leaf=1,      # Minimum number of samples required to be at a leaf node
        random_state=42          # Random state for reproducibility
    )
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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Get column names from the dataframe
    columns = df.columns.tolist()

    # Convert dataframe to HTML table
    table_html = df.head(5).to_html(classes='table table-striped', index=False)

    return jsonify({'columns': columns, 'table_html': table_html})

@app.route('/predict', methods=['POST'])
def make_prediction():
    file = request.files['file']
    df = pd.read_csv(file)
    target_column = request.form['target_column']
    
    # Preprocess the data
    X, y = preprocess_data(df, target_column)

    # Make prediction
    prediction = predict(df, target_column, X)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
