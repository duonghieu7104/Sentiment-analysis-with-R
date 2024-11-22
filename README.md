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
- [10. Demo](#10-demo)
- [11. Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

## 1. Thu thập dữ liệu từ Reddit API
- Sử dụng RedditExtractor https://github.com/ivan-rivera/RedditExtractor (Crawl Reddit API)
- Tải package
```r
install.packages("RedditExtractoR") 
```
```r
library(RedditExtractoR)

crawl_reddit_comments(
  keyword = "key-word", # "review", "game"
  num_threads = 10,
   min_comments = 10,
   date_from = "yyyy-mm-dd", # "2024-01-01" 
   date_to = "yyyy-mm-dd", # "2024-11-16"
   sort_by = "top",
   output_file = "your-file.csv"
 )
```

- Thu thập bài viết và bình luận từ các subreddit, key word
- Lưu trữ dữ liệu thô vào định dạng CSV

## 2. Gắn nhãn với Gemini API
- Thiết lập và cấu hình Gemini API
- Xây dựng prompt template cho việc gắn nhãn
```
"Given a sentence, classify its emotional tone into one of the following six emotions: sadness (0), joy (1), love (2), anger (3), fear (4), or surprise (5). Return only the corresponding ID of the detected emotion.
  Examples:
  Input: I can\'t believe this happened, I am so frustrated!
  Output: 3
  Input: I am so proud of you. I love you so much.
  Output: 2
  Input: This is amazing news! I am overjoyed!
  Output: 1
  Input: I am scared of what might happen next.
  Output: 4
  Task: Classify the emotion of the following sentence and return only the ID:
  %s"
```
- Xử lý và gắn nhãn tự động cho dữ liệu
```
body <- list(
    contents = list(
      list(
        parts = list(
          list(text = prompt)
        )
      )
    )
  )
  
  response <- POST(
    url = paste0(url, "?key=", api_key),
    body = toJSON(body, auto_unbox = TRUE),
    add_headers("Content-Type" = "application/json"),
    encode = "json"
  )
  
raw_content <- rawToChar(response$content)

cat("1. Raw Response:", raw_content, "\n\n")

tryCatch({
  content <- fromJSON(raw_content)
  cat("2. Parsed content structure:\n")
  str(content)
  cat("\n")
  
  candidates <- content$candidates
  cat("3. Candidates structure:\n")
  str(candidates)
  cat("\n")
  
  if (!is.null(candidates) && length(candidates) > 0) {
    response_text <- candidates$content$parts[[1]]$text
    cat("4. Raw text value:", response_text, "\n")
    
    cleaned_text <- trimws(response_text)
    cat("5. Cleaned text value:", cleaned_text, "\n")
    
    emotion_id <- as.numeric(cleaned_text)
    cat("6. Final numeric value:", emotion_id, "\n")
    
    if (!is.na(emotion_id)) {
      return(emotion_id)
    } else {
      cat("Error: Không thể chuyển đổi thành số.\n")
      return(NULL)
    }
  } else {
    cat("Error: Không tìm thấy candidates.\n")
    return(NULL)
  }
}, error = function(e) {
  cat("Error processing response:", e$message, "\n")
  return(NULL)
  })
}
```
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
- Tải mô hình Word2Vec đã được huấn luyện trước trên bộ dữ liệu Google News 300.
- Áp dụng mô hình để chuyển đổi các từ trong tập dữ liệu thành các vector embedding.
- Tạo document embedding
- Lưu trữ embedding vectors

## 7. Huấn luyện Random Forest
- Sử dụng thư viện library(ranger)
  ```r
  library(ranger)
  ```
  Train model và lưu model
  ```r
  model_rf <- ranger(
  label ~ ., 
  data = data_train, 
  num.trees = 500,
  probability = TRUE,
  importance = 'impurity'
  )
  saveRDS(model_rf, "sentiment_model.rds")
  ```

## 8. Huấn luyện XGBoost


## 9. Huấn luyện Neural Network

## 10. Demo

- Tải model
  ```r
  vocab <- readRDS("path-to-your-vocab")
  vectorizer <- vocab_vectorizer(vocab)
  model <- readRDS("path-to-your-model")
  ```

### Ví dụ 1: Vẽ biểu đồ xác suất cảm xúc khi dự đoán 1 câu
```r
plot_emotion_probabilities("I am afraid to face this", model, vectorizer)
```
![image](https://github.com/user-attachments/assets/b2e011ef-69d4-4b30-ab55-0c9dbb46345b)

### Ví dụ 2: Vẽ biểu đồ số lượng cảm xúc trong 1 file csv
```r
result <- analyze_emotions("path-to-your-file-csv", model, vectorizer)
print(result$emotion_counts)
result$plot
```
![image](https://github.com/user-attachments/assets/0f92ffd9-655c-4682-93a6-ed3a12abe356)


## 11. Tài liệu tham khảo

1. **Data Science Your Way**  
   - Repository hướng dẫn thực hiện phân tích cảm xúc.  
   - [GitHub Link](https://github.com/jadianes/data-science-your-way/tree/master/04-sentiment-analysis)

2. **Visualization and Sentiment Analysis**  
   - Phân tích và trực quan hóa cảm xúc.  
   - [Kaggle Link](https://www.kaggle.com/code/shaliniyaramada/visualization-and-sentiment-analysis)

3. **RedditExtractor**  
   - Công cụ thu thập dữ liệu từ Reddit.  
   - [GitHub Link](https://github.com/ivan-rivera/RedditExtractor)

4. **Word2Vec GoogleNews Vectors**  
   - Mô hình Word2Vec.  
   - [GitHub Link](https://github.com/mmihaltz/word2vec-GoogleNews-vectors)

5. **GoogleNews-vectors-negative300 (Word2Vec)**  
   - Bộ dữ liệu Word2Vec.  
   - [Kaggle Link](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)


## License
[MIT License](LICENSE)
