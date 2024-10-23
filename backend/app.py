from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import os

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'))

df_knn = data[['category_name', 'author', 'stars', 'price']].copy()

label_encoder_category = LabelEncoder()
df_knn['category_encoded'] = label_encoder_category.fit_transform(df_knn['category_name'])

label_encoder_author = LabelEncoder()
df_knn['author_encoded'] = label_encoder_author.fit_transform(df_knn['author'])

scaler = MinMaxScaler()
df_knn[['stars_normalized', 'price_normalized']] = scaler.fit_transform(df_knn[['stars', 'price']])

X = df_knn[['category_encoded', 'author_encoded', 'stars_normalized', 'price_normalized']]
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X)

app = Flask(__name__, template_folder='../templates', static_folder='../css')

@app.route('/', methods=['GET'])
def index():
    search_query = request.args.get('search')
    
    if search_query:
        books = data[data['title'].str.contains(search_query, case=False, na=False)][['title', 'imgUrl', 'category_name', 'author', 'price', 'stars']].to_dict(orient='records')
    else:
        books = data[['title', 'imgUrl', 'category_name', 'author', 'price', 'stars']].to_dict(orient='records')
    
    return render_template('index.html', books=books)

@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['title']
    
    selected_book_index = data[data['title'] == book_title].index[0]
    
    selected_book_features = X.iloc[selected_book_index].values.reshape(1, -1)
    distances, indices = knn.kneighbors(selected_book_features)

    recommended_books = data.iloc[indices[0]][['title', 'imgUrl', 'category_name', 'author', 'price', 'stars']].to_dict(orient='records')
    
    recommended_books = [book for book in recommended_books if book['title'] != book_title]

    average_distance = distances[0][1:].mean() if len(recommended_books) > 0 and len(distances[0]) > 1 else None

    return render_template('recommendations.html', books=recommended_books, average_distance=average_distance)

if __name__ == '__main__':
    app.run(debug=True)
