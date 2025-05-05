import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(csv_data_dir):
    train_df = pd.read_csv(csv_data_dir / 'train_index.csv')
    val_df = pd.read_csv(csv_data_dir / 'val_index.csv')
    test_df = pd.read_csv(csv_data_dir / 'test_index.csv')
    return train_df, val_df, test_df


def prepare_data(df):
    # Add `country_id` and one-hot encode it
    categorical_features = ['country_id']
    X_categorical = pd.get_dummies(df[categorical_features], drop_first=True)
    
    # Select numeric features
    numeric_features = ['latitude', 'longitude', 'elevation_m', 'humidity_pct', 'ndvi', 'night_lights', 'pop_density', 'slope_deg', 'soil_moisture', 'temperature_c']
    X_numeric = df[numeric_features]
    
    # Combine numeric and one-hot encoded categorical features
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df['label']
    return X, y, numeric_features


def plot_confusion_matrix(y_true, y_pred, classes, model_type, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))  
    sns.heatmap(cm, annot=True, fmt='.2f', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_type}: Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')  
    plt.yticks(rotation=0)  
    save_path = Path(save_path)
    plt.tight_layout()  
    plt.savefig(save_path / f'{model_type}_confusion.png')
    plt.close()


def main(data_dir, use_cv):
    csv_data_dir = Path(data_dir) / 'csv_data'
    train_df, val_df, test_df = load_data(csv_data_dir)

    # Combine train and validation sets
    combined_train_df = pd.concat([train_df, val_df])

    X_train, y_train, numeric_features = prepare_data(combined_train_df)
    X_test, y_test, _ = prepare_data(test_df)

    # Scale only numeric data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numeric_features])
    X_test_scaled = scaler.transform(X_test[numeric_features])

    # Append categorical data back to the scaled numeric data
    X_train_scaled = np.hstack((X_train_scaled, X_train.drop(columns=numeric_features).values))
    X_test_scaled = np.hstack((X_test_scaled, X_test.drop(columns=numeric_features).values))

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=5000, solver='saga', C=1),
        'kNN': KNeighborsClassifier(n_neighbors=3),
        'SVM': SVC(C=0.1, kernel='linear'),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2)
    }

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1]},
        'kNN': {'n_neighbors': [3, 5, 7]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'Random Forest': {'n_estimators': [200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
    }

    best_models = {}

    # Train models
    for name, model in models.items():
        if use_cv:
            print(f'Tuning {name} with cross-validation...')
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy')
            grid_search.fit(X_train_scaled, y_train)
            best_models[name] = grid_search.best_estimator_
            print(f'Best parameters for {name}: {grid_search.best_params_}')
        else:
            print(f'Training {name} with default parameters...')
            model.fit(X_train_scaled, y_train)
            best_models[name] = model

    # Evaluate on test set
    for name, model in best_models.items():
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'{name} Test Accuracy: {accuracy:.4f}')

        # Plot confusion matrix
        classes = y_test.unique()
        plot_confusion_matrix(y_test, y_pred, classes, model_type=name, save_path="../results")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train simple models on non-image data.')
    parser.add_argument('-d', '--data-dir', type=str, required=True, help='Directory where the CSV data is stored')
    parser.add_argument('--use-cv', action='store_true', help='Use cross-validation for hyperparameter tuning')
    args = parser.parse_args()

    main(args.data_dir, args.use_cv)
