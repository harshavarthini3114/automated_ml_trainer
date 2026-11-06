import pandas as pd  # pyright: ignore[reportMissingModuleSource]

from sklearn.preprocessing import StandardScaler, OneHotEncoder  # pyright: ignore[reportMissingModuleSource]
from sklearn.compose import ColumnTransformer  # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingModuleSource]

from src.db_loader import stream_joined_records, get_column_names
from src.logger import get_production_logger


class DataPipeline:

    def __init__(self):
        self.logger = get_production_logger()
        self.df = None
        self.X = None
        self.y = None
        self.numerical_columns = None
        self.categorical_columns = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data_from_warehouse(self):
        self.logger.info("Data loading started")
        records = list(stream_joined_records())
        column = get_column_names()
        df = pd.DataFrame(records, columns=column)
        self.df = df

        self.logger.info(f"Loaded {len(df)} records")
        return df

    def explore_data(self):
        if self.df is not None:
            self.logger.info("data has loaded")
        else:
            self.logger.info("data isn't load")
            return

        self.logger.info("Data exploration started")
        self.logger.info(f"Dataset shape {self.df.shape}")
        self.logger.info(f"First five rows:\n{self.df.head()}")
        self.logger.info(f"Column data types:\n{self.df.dtypes}")
        self.logger.info(f"Missing values:\n{self.df.isnull().sum()}")

    def separate_features_target(self):
        X = self.df.drop(columns=["customerID", "Churn"])
        y = self.df["Churn"]

        self.X = X
        self.y = y

        self.logger.info(f"shape of x {self.X.shape}")
        self.logger.info(f"shape of y {self.y.shape}")

    def identify_column_types(self):
        # "number" covers int32/int64/float64 from SQLite → pandas without a schema
        numerical_columns = self.X.select_dtypes(include=["number"]).columns
        categorical_columns = self.X.select_dtypes(include=["object"]).columns

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        self.logger.info(f"numerical columns {self.numerical_columns}")
        self.logger.info(f"categorical columns {self.categorical_columns}")

    def create_preprocessor(self):
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        # Unfitted blueprint for sklearn Pipeline in ModelEngine (no leakage across CV folds)
        self.preprocessor = ColumnTransformer([
            ("num", numerical_transformer, self.numerical_columns),
            ("cat", categorical_transformer, self.categorical_columns)
        ])
        return self.preprocessor

    def split_and_process(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.logger.info(f"shape of x_train {self.X_train.shape}")
        self.logger.info(f"shape of x_test {self.X_test.shape}")
        self.logger.info(f"shape of y_train {self.y_train.shape}")
        self.logger.info(f"shape of y_test {self.y_test.shape}")

    def get_processed_data(self):
        return self.preprocessor, self.X_train, self.X_test, self.y_train, self.y_test
