from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import os
import webbrowser
import threading

data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'))

df_knn = data[['category_name', 'author', 'stars', 'price']].copy()

label_encoder_category = LabelEncoder()
df_knn['category_encoded'] = label_encoder_category.fit_transform(df_knn['category_name'])

scaler = MinMaxScaler()
df_knn[['stars_normalized', 'price_normalized']] = scaler.fit_transform(df_knn[['stars', 'price']])

X = df_knn[['category_encoded', 'stars_normalized', 'price_normalized']]
knn = NearestNeighbors(n_neighbors=6, algorithm='auto', metric='euclidean')
knn.fit(X)

app = Flask(__name__, template_folder='../views', static_folder='../css')

BOOKS_PER_PAGE = 100

@app.route('/', methods=['GET'])
def index():
    search_query = request.args.get('search')
    page = int(request.args.get('page', 1))

    if search_query:
        filtered_books = data[data['title'].str.contains(search_query, case=False, na=False, regex=False)]
    else:
        filtered_books = data

    total_books = filtered_books.shape[0]
    total_pages = (total_books + BOOKS_PER_PAGE - 1) // BOOKS_PER_PAGE
    start_index = (page - 1) * BOOKS_PER_PAGE
    end_index = start_index + BOOKS_PER_PAGE
    books = filtered_books[start_index:end_index][['title', 'imgUrl', 'category_name', 'author', 'price', 'stars']].to_dict(orient='records')

    pagination_start = max(1, page - 2)
    pagination_end = min(total_pages, page + 2)

    return render_template(
        'index.html',
        books=books,
        total_pages=total_pages,
        current_page=page,
        pagination_start=pagination_start,
        pagination_end=pagination_end
    )

@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['title']
    
    if book_title not in data['title'].values:
        return render_template('recommendations.html', selected_book=book_title, books=[], average_distance=None)

    selected_book_index = data[data['title'] == book_title].index[0]
    selected_book_features = X.iloc[selected_book_index].values.reshape(1, -1)
    distances, indices = knn.kneighbors(selected_book_features)

    recommended_books = data.iloc[indices[0]][['title', 'imgUrl', 'category_name', 'author', 'price', 'stars']].to_dict(orient='records')

    distance_limit = 1.0
    filtered_books = [
        book for i, book in enumerate(recommended_books) 
        if distances[0][i] <= distance_limit and book['title'] != book_title
    ]

    if len(filtered_books) < 5:
        additional_books = [
            book for book in recommended_books 
            if book['title'] != book_title and book not in filtered_books
        ]
        filtered_books.extend(additional_books[:5 - len(filtered_books)])

    filtered_books = filtered_books[:5]

    if len(filtered_books) > 0:
        distances_for_average = distances[0][1:len(filtered_books) + 1] 
        average_distance = distances_for_average.mean() if len(distances_for_average) > 0 else None
    else:
        average_distance = None

    selected_book_info = {
        'title': book_title,
        'imgUrl': data.loc[selected_book_index, 'imgUrl'],
        'category_name': data.loc[selected_book_index, 'category_name'],
        'author': data.loc[selected_book_index, 'author'],
        'price': data.loc[selected_book_index, 'price'],
        'stars': data.loc[selected_book_index, 'stars']
    }

    return render_template('recommendations.html', selected_book=selected_book_info, books=filtered_books, average_distance=average_distance)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search_query = request.args.get('query', '')
    if search_query:
        suggestions = data[data['title'].str.contains(search_query, case=False, na=False)]
        suggestions = suggestions['title'].tolist()
    else:
        suggestions = []
    
    return jsonify(suggestions)

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=True)