import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample

class KNNClassifierTester:
    def __init__(self, df_meta, df_dists, class_type=int):
        self.df_meta = df_meta
        self.df_dists = df_dists
        self.class_type = class_type
        self.classes = np.unique(list(df_meta['Class']))

    def knn_bootstrap(self, n_bootstrapping=3, training_set_size=15, training_proportion=0.35):
        all_classification_results = []

        for _ in range(n_bootstrapping):
            # Randomly select training samples using bootstrapping
            training_samples = resample(self.df_meta, n_samples=training_set_size, replace=True, random_state=None)
            testing_samples = self.df_meta.loc[~self.df_meta.index.isin(training_samples.index)]

            # Assuming your features are stored in df_dists, and labels in df_meta['Class']
            X_train = self.df_dists.loc[training_samples.index]
            y_train = training_samples['Class'].astype(self.class_type)

            X_test = self.df_dists.loc[testing_samples.index]
            y_test = testing_samples['Class'].astype(self.class_type)

            # Create and train KNN classifier
            knn_classifier = KNeighborsClassifier(n_neighbors=3)
            knn_classifier.fit(X_train, y_train)

            # Make predictions
            y_pred = knn_classifier.predict(X_test)

            # Calculate F1 score
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Save classification results
            all_classification_results.append({'F1 Score': f1, 'Predictions': y_pred, 'True Labels': y_test})

        return all_classification_results

# Example usage:
df_meta = pd.read_csv()
df_dists = pd.read_csv()

# Create an instance of KNNClassifierTester
knn_tester = KNNClassifierTester(df_meta, df_dists, class_type=int)

# Run KNN with bootstrapping
results = knn_tester.knn_bootstrap(n_bootstrapping=3, training_set_size=15, training_proportion=0.35)

# Access the results as needed
for idx, result in enumerate(results):
    print(f"Bootstrap Iteration {idx + 1}: F1 Score = {result['F1 Score']}")

# You can also access predictions and true labels for each iteration
# For example, predictions of the first iteration:
print("Predictions (Iteration 1):", results[0]['Predictions'])
print("True Labels (Iteration 1):", results[0]['True Labels'])

