# Dự án Phân tích Dữ liệu Reddit với Machine Learning

## Mục lục
- [1. Thu thập dữ liệu từ Reddit API](#1-thu-thập-dữ-liệu-từ-reddit-api)
- [2. Gắn nhãn với Gemini API](#2-gắn-nhãn-với-gemini-api)
- [3. Tiền xử lý dữ liệu](#3-tiền-xử-lý-dữ-liệu)
- [4. Trực quan hóa dữ liệu](#4-trực-quan-hóa-dữ-liệu)
- [5. Tạo TF-IDF](#5-tạo-tf-idf)
- [6. Tạo Embedding](#6-tạo-embedding)
- [7. Huấn luyện Random Forest](#7-huấn-luyện-random-forest)
- [8. Huấn luyện XGBoost](#8-huấn-luyện-xgboost)
- [9. Huấn luyện Neural Network](#9-huấn-luyện-neural-network)

## 1. Thu thập dữ liệu từ Reddit API
- Sử dụng PRAW (Python Reddit API Wrapper) để thu thập dữ liệu
- Cấu hình authentication với Reddit API
- Thu thập bài viết và bình luận từ các subreddit
- Lưu trữ dữ liệu thô vào định dạng JSON/CSV

## 2. Gắn nhãn với Gemini API
- Thiết lập và cấu hình Gemini API
- Xây dựng prompt template cho việc gắn nhãn
- Xử lý và gắn nhãn tự động cho dữ liệu
- Kiểm tra chất lượng gắn nhãn
- Xuất dữ liệu đã được gắn nhãn

## 3. Tiền xử lý dữ liệu
- Làm sạch văn bản (xóa emoji, ký tự đặc biệt)
- Chuẩn hóa text (lowercase, loại bỏ dấu câu)
- Tokenization
- Loại bỏ stopwords
- Lemmatization/Stemming

## 4. Trực quan hóa dữ liệu
- Phân tích phân bố các nhãn
- Visualization độ dài văn bản
- Word clouds
- Biểu đồ tần suất từ


## 5. Tạo TF-IDF
- Xây dựng vocabulary
- Tính toán term frequency (TF)
- Tính toán inverse document frequency (IDF)
- Tạo ma trận TF-IDF
- Lưu trữ và normalize ma trận

## 6. Tạo Embedding
- Sử dụng pre-trained models (Word2Vec/FastText)
- Tạo document embedding
- Lưu trữ embedding vectors

## 7. Huấn luyện Random Forest


## 8. Huấn luyện XGBoost


## 9. Huấn luyện Neural Network




## License
[MIT License](LICENSE)
