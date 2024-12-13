import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
import re
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# Load dataset
data = pd.read_csv('hci_final_dataset.csv')

# Encode labels
label_encoder = LabelEncoder()
data['status_encoded'] = label_encoder.fit_transform(data['status'])

X = data.drop(columns=['url', 'status', 'status_encoded'])
y = data['status_encoded']
feature_columns = X.columns.tolist()

# XGBoost parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 13,
    "eta": 0.05056910450877249,
    "subsample": 0.931847480641822,
    "colsample_bytree": 0.6534069655471098,
    "gamma": 0.3983230621965699,
    "min_child_weight": 1,
    "lambda": 1.1981000858663946,
    "alpha": 0.1565955240203879,
    "seed": 42
}
num_boost_round = 686

# Train models
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
models = []

for train_idx, test_idx in skf.split(X, y):
    dtrain = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)
    models.append(model)

def extract_features_from_url(url):
    features = {
        "length_url": len(url),
        "length_hostname": len(urlparse(url).hostname) if urlparse(url).hostname else 0,
        "ip": 1 if any(char.isdigit() for char in urlparse(url).hostname) else 0,
        "nb_dots": url.count('.'),
        "nb_hyphens": url.count('-'),
        "nb_at": url.count('@'),
        "nb_qm": url.count('?'),
        "nb_and": url.count('&'),
        "nb_or": url.count('|'),
        "nb_eq": url.count('='),
        "nb_underscore": url.count('_'),
        "nb_tilde": url.count('~'),
        "nb_percent": url.count('%'),
        "nb_slash": url.count('/'),
        "nb_star": url.count('*'),
        "nb_colon": url.count(':'),
        "nb_comma": url.count(','),
        "nb_semicolumn": url.count(';'),
        "nb_dollar": url.count('$'),
        "nb_space": url.count(' '),
        "nb_www": 1 if "www" in url else 0,
        "nb_com": 1 if ".com" in url else 0,
        "nb_dslash": url.count('//'),
        "http_in_path": 1 if "http" in urlparse(url).path else 0,
        "https_token": 1 if "https" in url else 0,
        "ratio_digits_url": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "ratio_digits_host": sum(c.isdigit() for c in urlparse(url).hostname) / len(urlparse(url).hostname) if urlparse(url).hostname else 0,
        "punycode": 1 if re.search(r'xn--', url) else 0,
        "port": urlparse(url).port if urlparse(url).port else 0,
        "tld_in_path": 1 if any(tld in urlparse(url).path for tld in ['.com', '.net', '.org', '.gov', '.edu']) else 0,
        "tld_in_subdomain": 1 if any(tld in urlparse(url).hostname for tld in ['.com', '.net', '.org', '.gov', '.edu']) else 0,
        "abnormal_subdomain": 1 if len(urlparse(url).hostname.split('.')) > 3 else 0,
        "nb_subdomains": len(urlparse(url).hostname.split('.')) - 1,
        "prefix_suffix": 1 if url.startswith("www") else 0,
        "shortening_service": 1 if any(short in url for short in ['bit.ly', 'goo.gl', 'tinyurl.com']) else 0,
        "path_extension": 1 if any(ext in urlparse(url).path for ext in ['.exe', '.zip', '.rar', '.tar', '.pdf']) else 0,
        "length_words_raw": len(url.split()),
        "char_repeat": len(set(url)),
        "shortest_words_raw": min(len(word) for word in url.split()) if url.split() else 0,
        "longest_words_raw": max(len(word) for word in url.split()) if url.split() else 0,
        "shortest_word_host": min(len(word) for word in urlparse(url).hostname.split('.')) if urlparse(url).hostname else 0,
        "longest_word_host": max(len(word) for word in urlparse(url).hostname.split('.')) if urlparse(url).hostname else 0,
        "shortest_word_path": min(len(word) for word in urlparse(url).path.split('/')) if urlparse(url).path else 0,
        "longest_word_path": max(len(word) for word in urlparse(url).path.split('/')) if urlparse(url).path else 0,
        "avg_words_raw": np.mean([len(word) for word in url.split()]) if url.split() else 0,
        "avg_word_host": np.mean([len(word) for word in urlparse(url).hostname.split('.')]) if urlparse(url).hostname else 0,
        "avg_word_path": np.mean([len(word) for word in urlparse(url).path.split('/')]) if urlparse(url).path else 0,
        "phish_hints": 1 if 'login' in url or 'secure' in url else 0,
        "domain_in_brand": 1 if 'apple' in urlparse(url).hostname else 0,
        "brand_in_subdomain": 1 if 'apple' in urlparse(url).hostname.split('.')[0] else 0,
        "brand_in_path": 1 if 'apple' in urlparse(url).path else 0,
        "suspicious_tld": 1 if urlparse(url).hostname.endswith(('.xyz', '.top', '.club')) else 0,
        "entropy": -sum([url.count(c) / len(url) * np.log2(url.count(c) / len(url)) for c in set(url)]) if len(url) > 0 else 0
    }

    features_df = pd.DataFrame([features])[feature_columns]
    return features_df

# Ensemble prediction
def predict_with_ensemble(models, url_features):
    predictions = np.mean([model.predict(xgb.DMatrix(url_features)) for model in models], axis=0)
    return predictions > 0.5

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/')
def home():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict_url():
    try:
        data = request.json
        url = data.get("url", "")
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        # Validate URL
        if not (url.startswith("https://") or url.startswith("http://")):
            return jsonify({"error": "Invalid URL. Must start with http or https."}), 400

        # Extract features and predict
        url_features = extract_features_from_url(url)
        
        if url_features.empty:
            raise ValueError("Extracted features are empty.")
        
        is_phishing = predict_with_ensemble(models, url_features)[0]
        result = "phishing" if is_phishing else "safe"

        return jsonify({"url": url, "result": result})
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error to the console
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's default port or fallback to 10000
    app.run(host="0.0.0.0", port=port, debug=True)


