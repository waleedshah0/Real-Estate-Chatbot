from flask import Flask, request, jsonify, render_template
import os
import zipfile
import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Ensure the ZIP file exists
zip_path = 'archive.zip'
extracted_path = 'extracted_data'

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"ZIP file '{zip_path}' not found!")

# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Locate the CSV file
csv_files = [f for f in os.listdir(extracted_path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the extracted data.")
csv_file = csv_files[0]
data_path = os.path.join(extracted_path, csv_file)

# Load the dataset
real_estate_data = pd.read_csv(data_path)

# Ensure required columns are present
required_columns = ['price', 'street', 'city', 'state', 'bed', 'bath']
missing_columns = [col for col in required_columns if col not in real_estate_data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in dataset: {missing_columns}")

# Data preprocessing function
def preprocess_data(df):
    df = df.drop_duplicates()
    df['price'] = np.where(df['price'] > df['price'].quantile(0.99), df['price'].quantile(0.99), df['price'])
    df['price'] = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
    return df

real_estate_data = preprocess_data(real_estate_data)

# Categorize price into bins
bins = [0, 0.2, 0.6, 1.0]
categories = ['Affordable', 'Standard', 'Luxury']
real_estate_data['Price Category'] = pd.cut(real_estate_data['price'], bins=bins, labels=categories, right=False)

# Sample queries and labels for model training
queries = [
    "I want a house in Puerto Rico under 200000 USD",
    "Looking for a house with 3 bedrooms under 500000 USD",
    "Suggest a luxury house in California",
    "Show me affordable houses in Texas",
    "I need a house with 2 bathrooms under 300000 USD",
    "Find me a standard house in Florida",
]
labels = ['Affordable', 'Standard', 'Luxury', 'Affordable', 'Standard', 'Standard']

# Tokenize queries
tokenizer = Tokenizer()
tokenizer.fit_on_texts(queries)
sequences = tokenizer.texts_to_sequences(queries)
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
label_map = {'Affordable': 0, 'Standard': 1, 'Luxury': 2}
y = np.array([label_map[label] for label in labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=max_len),
    LSTM(16),
    Dense(3, activation='softmax')
])
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=1)

# Predict on test data
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Recommendation function (only top 3 results)
def recommend_houses(user_query):
    seq = tokenizer.texts_to_sequences([user_query])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded_seq)
    category_idx = np.argmax(pred)
    category_map = {0: 'Affordable', 1: 'Standard', 2: 'Luxury'}
    category = category_map[category_idx]

    state = None
    budget = None
    bed_count = None

    # Parse the user query for state, budget, and bedrooms
    words = user_query.lower().split()
    for word in words:
        if word.title() in real_estate_data['state'].unique():
            state = word.title()
        if word.isdigit():
            budget = int(word)
        if 'bed' in word:
            bed_match = re.search(r'(\d+)\s*bed', word)
            if bed_match:
                bed_count = int(bed_match.group(1))

    # Filter data based on query
    filtered_data = real_estate_data
    if category:
        filtered_data = filtered_data[filtered_data['Price Category'] == category]
    if state:
        filtered_data = filtered_data[filtered_data['state'] == state]
    if budget:
        filtered_data = filtered_data[filtered_data['price'] <= budget]
    if bed_count:
        filtered_data = filtered_data[filtered_data['bed'] == bed_count]

    # Limit to top 3 results
    filtered_data = filtered_data.head(3)

    if filtered_data.empty:
        return real_estate_data[['street', 'city', 'state', 'price', 'bed', 'bath', 'Price Category']].head(3)

    return filtered_data[['street', 'city', 'state', 'price', 'bed', 'bath', 'Price Category']]

# Flask routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_query = data.get("query", "")
    recommendations = recommend_houses(user_query)
    if recommendations.empty:
        return jsonify({"message": "No matching houses found."})
    return recommendations.to_json(orient="records")

if __name__ == "__main__":
    app.run(debug=True)
