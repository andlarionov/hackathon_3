import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd
import wget
import re
import os

def get_df():
    # Скачивание файла
    url = 'https://github.com/yandex/geo-reviews-dataset-2023/raw/master/geo-reviews-dataset-2023.tskv'
    file_path = 'geo-reviews-dataset-2023.tskv'
    if not os.path.exists(file_path):
        # Скачивание файла, если он не существует
        wget.download(url, file_path)
    
    # Список для хранения данных
    data = []
    # Чтение файла построчно
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Удаляем пробелы и символы новой строки
            line = line.strip()
            if line:  # Проверяем, что строка не пустая
                # Разделяем строку на пары "ключ=значение"
                items = line.split('\t')  # tskv использует табуляцию как разделитель
                data_dict = {}
                for item in items:
                    key, value = item.split('=', 1)  # Разделяем только по первому '='
                    data_dict[key] = value
                data.append(data_dict)

    # Создание DataFrame из списка словарей
    df = pd.DataFrame(data)
    # Разделение рубрик по разделителю и создание новых строк
    df['rubrics'] = df['rubrics'].str.replace(';', ',')  # Сначала заменяем ';' на ','
    df['rubrics'] = df['rubrics'].str.split(',')  # Теперь разбиваем по ','
    df = df.explode('rubrics')
    df.rubrics.value_counts()
    # Поиск дубликатов
    duplicates = df.duplicated(subset=['text'])
    num_duplicates = duplicates.sum()
    df = df.drop_duplicates(subset=['text'])
    df['rating'] = df['rating'].str.replace('.', '', regex=False).astype(int)
    # Очистка текста
    def clear_text(text):
        # Удаление хэштегов
        text = re.sub(r"#(\w+)", r"\1", text)
        # Удаление эмодзи, сохраняя цифры, знаки препинания и знаки плюс и минус
        text = re.sub(r"[^\w\s,.!?;:0-9+-]", "", text)  # Сохраняем + и - для акций и диапазонов
        # Удаляем все символы n и любые другие нежелательные символы
        text = re.sub(r'n', '', text)
        # Удаление лишних пробелов
        text = " ".join(text.split())
        return text
    # Очистка и лемматизация
    df['clear_text'] = df['text'].apply(clear_text)
    return df

@st.cache_data
def load_data():
    return get_df()

# Кэширование загрузки модели spaCy
@st.cache_resource
def load_spacy_model():
    return spacy.load("ru_core_news_sm")

df = load_data()
nlp = load_spacy_model()


# Пример использования загруженной модели и объектов
def tokenize_by_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return tokens

# Функция для извлечения резюме
def extractive_summary(text, model, top_n=5):
    doc = nlp(text)  # Используем spaCy для обработки текста
    sentences = [sent.text for sent in doc.sents]  # Разбиение текста на предложения с помощью spaCy
    tokenized_sentences = [tokenize_by_spacy(sentence) for sentence in sentences]

    # Получение векторов для всех предложений
    sentence_vectors = np.array([get_sentence_vector(sentence, model) for sentence in tokenized_sentences])

    # Вычисление матрицы схожести между предложениями
    similarity_matrix = cosine_similarity(sentence_vectors)

    # Ранжирование предложений по значимости
    scores = similarity_matrix.sum(axis=1)

    # Получение индексов наиболее значимых предложений
    ranked_indices = np.argsort(scores)[-top_n:][::-1]

    # Формирование итогового резюме
    summary = [sentences[i] for i in ranked_indices]

    return summary

# Функция получения векторного представления предложения
def get_sentence_vector(sentence, model):
    vector = np.zeros(model.vector_size)
    count = 0

    for word in sentence:
        if word in model.wv:
            vector += model.wv[word]
            count += 1

    if count > 0:
        vector /= count
    return vector



def search_organizations(df, city_name, street_name, rubric_name, min_rating):

    result_str = ""

    matching_rows = df[df['address'].str.contains(city_name, case=False, na=False)]

    if matching_rows.empty:
        return (f"Город '{city_name}' не найден в адресах.")
        

    street_filtered = matching_rows[matching_rows['address'].str.contains(street_name, case=False, na=False)]

    if street_filtered.empty:
        return (f"В городе '{city_name}' нет организаций на улице '{street_name}'.")
        

    rubric_filtered = street_filtered[street_filtered['rubrics'].str.contains(rubric_name, case=False, na=False)]

    if rubric_filtered.empty:
        return (f"В городе '{city_name}' на улице '{street_name}' нет организаций в рубрике '{rubric_name}'.")
        

    final_results = rubric_filtered[rubric_filtered['rating'] >= min_rating]

    if final_results.empty:
        return (f"В городе '{city_name}' на улице '{street_name}' нет организаций в рубрике '{rubric_name}' с рейтингом не ниже {min_rating}.")
        

    result_names = final_results['name_ru'].unique()
    result_str = (f"Найденные организации {len(result_names)}: \n{', '.join(result_names)}")
    word2vec_model1 = joblib.load('word2vec_model.pkl')
    summaries = {}
    for name in result_names:
        supplier_reviews = final_results[final_results['name_ru'] == name]['clear_text'].tolist()
        combined_reviews_text = ' '.join(supplier_reviews)
        
        summary = extractive_summary(combined_reviews_text, word2vec_model1)
        summaries[name] = summary

    result_str += "\n"
    result_str += "\nРезюме по отзывам:\n"
    for supplier, summary in summaries.items():
        result_str += "\n"
        result_str += (f"{supplier}: {' '.join(summary)} \n")

    print(result_str)
    return result_str

def get_reviews_for_restaurant(df, restaurant_name):

    # Фильтрация отзывов по названию ресторана
    reviews = df[df['name_ru'] == restaurant_name]['clear_text']
    # Объединение всех отзывов в одну строку
    combined_reviews = ' '.join(reviews)
    return combined_reviews


def get_city():

    def extract_city(address):
        # Разделяем адрес по запятым и берем первый элемент
        parts = address.split(',')
        return parts[0].strip()

    # Применяем функцию к столбцу address
    df['city'] = df['address'].apply(extract_city)

    # Подсчитываем количество каждого города
    return df['city'].value_counts().index.tolist()

def get_streets():

    # Функция для извлечения улицы из адреса
    def extract_street(address):
        # Разделяем адрес по запятым и берем второй элемент
        parts = address.split(',')
        if len(parts) > 1:
            return parts[1].strip()
        return None
    df['street'] = df['address'].apply(extract_street)

    return df['street'].unique().tolist()

def main():

    st.title("Приложение для генерации отзывов")
    citis = get_city()
    # Ввод данных от пользователя
    city = st.selectbox("Введите название города/субъекта РФ:", citis)
    street = st.text_input("Введите название улицы:")
    category = st.selectbox("Введите название рубрики:", ['Ресторан', 'Гостиница', 'Кафе', 'Супермаркет', 'Салон красоты', 'Магазин продуктов', 'Быстрое питание', 'Торговый центр', 'Музей', 'Бар'])
    rating = st.slider("Введите минимальный рейтинг (например, 4.0):", 0.0, 5.0, 3.0, step=0.1,)



    # Отображение найденных организаций
    if st.button("Генерировать отзывы"):
        try:
            st.write(search_organizations(df, city, street, category, rating))
        except KeyboardInterrupt:
            print("\nПрограмма завершена.")
        except Exception as e:
            print(f"Ошибка: {e}")
        




if __name__ == '__main__':
    main()