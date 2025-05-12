import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class ProfessionPredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.X = None
        self.y = None
        self._load_data()
        self._train_model()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        self.X = df[['average_score', 'favorite_subject']]
        self.y = df['profession']

    def _train_model(self):
        categorical_features = ['favorite_subject']
        numeric_features = ['average_score']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ])

        self.model.fit(self.X, self.y)

    def predict(self, average_score, favorite_subject):
        input_df = pd.DataFrame({
            'average_score': [average_score],
            'favorite_subject': [favorite_subject]
        })
        return self.model.predict(input_df)[0]
