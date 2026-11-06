from src.db_loader import initialize_data_warehouse
from src.data_pipeline import DataPipeline
from src.model_engine import ModelEngine


def main():
    initialize_data_warehouse()

    pipeline = DataPipeline()
    pipeline.load_data_from_warehouse()
    pipeline.explore_data()
    pipeline.separate_features_target()
    pipeline.identify_column_types()
    pipeline.create_preprocessor()
    pipeline.split_and_process()

    preprocessor, X_train, X_test, y_train, y_test = pipeline.get_processed_data()

    engine = ModelEngine()
    champion = engine.run(preprocessor, X_train, X_test, y_train, y_test)
    print(f"Champion: {champion}")


if __name__ == "__main__":
    main()