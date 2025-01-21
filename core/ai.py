import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class BaseForexPredictor:
    def __init__(self):
        self.event_scaler = StandardScaler()
        self.forex_scaler = StandardScaler()
        self.model = None

    def preprocess_events(self, events_df):
        """
        Przetwarza podstawowe cechy wydarzeń.

        Args:
            events_df (pd.DataFrame): DataFrame z wydarzeniami

        Returns:
            np.array: Przetworzone cechy wydarzeń
        """
        feature_columns = [
            "consensus_diff",
            "forecast_diff",
            "change",
            "actual_change_pct",
            "consensus_diff_pct",
            "forecast_diff_pct",
            "impact_range",
            "impact_range_pct",
        ]

        events_df[feature_columns] = events_df[feature_columns].fillna(0)
        numeric_features = self.event_scaler.fit_transform(events_df[feature_columns])

        title_vectors = np.array(
            [eval(str(vec)) for vec in events_df["title_code"].values]
        )

        return np.hstack((title_vectors, numeric_features))

    def create_base_model(self, input_dim, output_dim):
        """
        Tworzy podstawową architekturę modelu.

        Args:
            input_dim (int): Wymiar wejścia
            output_dim (int): Wymiar wyjścia

        """
        inputs = Input(shape=(input_dim,))

        x = Dense(256, activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        outputs = Dense(output_dim)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

        self.model = model

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Trenuje model.

        Args:
            X_train (np.array): Dane treningowe
            y_train (np.array): Etykiety treningowe
            epochs (int): Liczba epok
            batch_size (int): Rozmiar batcha
            validation_split (float): Część danych do walidacji

        Returns:
            History: Historia treningu
        """
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f"best_{self.__class__.__name__}.keras",
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        return self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
        )

    @classmethod
    def train_model(cls, events_df, forex_df, *args, **kwargs):
        model = cls()
        X_data, y_data = model.prepare_data(events_df, forex_df)
        input_dim = X_data.shape[1]
        output_dim = y_data.shape[1]
        model.create_base_model(input_dim, output_dim)
        return model.train(X_data, y_data)

class ShortTermForexPredictor(BaseForexPredictor):
    def __init__(self, time_window=30):
        """
        Inicjalizacja modelu krótkoterminowego.

        Args:
            time_window (int): Okno czasowe w minutach (przed i po wydarzeniu)
        """
        super().__init__()
        self.time_window = time_window

    def prepare_data(self, events_df, forex_df):
        """
        Przygotowuje dane do treningu modelu krótkoterminowego.

        Args:
            events_df (pd.DataFrame): DataFrame z wydarzeniami
            forex_df (pd.DataFrame): DataFrame z danymi forex (30min interwał)

        Returns:
            tuple: (X_train, y_train) - dane do treningu
        """
        events_df = events_df.copy()
        forex_df = forex_df.copy()

        events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
        forex_df["timestamp"] = pd.to_datetime(forex_df["timestamp"])

        X_features = []
        y_changes = []

        for idx, event in events_df.iterrows():
            relevant_pairs = forex_df[forex_df["base_currency"] == event["currency"]]

            event_time = event["timestamp"]
            before_window = event_time - pd.Timedelta(minutes=self.time_window)
            after_window = event_time + pd.Timedelta(minutes=self.time_window)

            for ticker, pair_data in relevant_pairs.groupby("ticker"):
                before_data = (
                    pair_data[
                        (pair_data["timestamp"] >= before_window)
                        & (pair_data["timestamp"] < event_time)
                    ].iloc[-1]
                    if len(pair_data) > 0
                    else None
                )

                after_data = (
                    pair_data[
                        (pair_data["timestamp"] > event_time)
                        & (pair_data["timestamp"] <= after_window)
                    ].iloc[0]
                    if len(pair_data) > 0
                    else None
                )

                if before_data is not None and after_data is not None:
                    changes = {
                        "open_change": (after_data["open"] - before_data["open"])
                        / before_data["open"],
                        "high_change": (after_data["high"] - before_data["high"])
                        / before_data["high"],
                        "low_change": (after_data["low"] - before_data["low"])
                        / before_data["low"],
                        "close_change": (after_data["close"] - before_data["close"])
                        / before_data["close"],
                    }

                    event_features = self.preprocess_events(pd.DataFrame([event]))
                    X_features.append(event_features[0])
                    y_changes.append(list(changes.values()))

        return np.array(X_features), np.array(y_changes)


class DailyForexPredictor(BaseForexPredictor):
    def prepare_data(self, events_df, forex_df):
        """
        Przygotowuje dane do treningu modelu dziennego.

        Args:
            events_df (pd.DataFrame): DataFrame z wydarzeniami
            forex_df (pd.DataFrame): DataFrame z dziennymi danymi forex

        Returns:
            tuple: (X_train, y_train) - dane do treningu
        """
        events_df = events_df.copy()
        forex_df = forex_df.copy()

        events_df["date"] = pd.to_datetime(events_df["timestamp"]).dt.date
        forex_df["date"] = pd.to_datetime(forex_df["timestamp"]).dt.date

        X_features = []
        y_changes = []

        for (date, currency), day_events in events_df.groupby(["date", "currency"]):
            day_forex = forex_df[
                (forex_df["date"] == date) & (forex_df["base_currency"] == currency)
            ]

            if len(day_forex) > 0:
                events_features = self.preprocess_events(day_events)
                avg_features = np.mean(events_features, axis=0)

                for ticker, pair_data in day_forex.groupby("ticker"):
                    changes = {
                        "open_change": (
                            pair_data["open"].iloc[-1] - pair_data["open"].iloc[0]
                        )
                        / pair_data["open"].iloc[0],
                        "high_change": (
                            pair_data["high"].max() - pair_data["high"].min()
                        )
                        / pair_data["high"].min(),
                        "low_change": (pair_data["low"].max() - pair_data["low"].min())
                        / pair_data["low"].min(),
                        "close_change": (
                            pair_data["close"].iloc[-1] - pair_data["close"].iloc[0]
                        )
                        / pair_data["close"].iloc[0],
                    }

                    X_features.append(avg_features)
                    y_changes.append(list(changes.values()))

        return np.array(X_features), np.array(y_changes)


def train_models(events_df, forex_df):
    """
    Trenuje oba modele.

    Args:
        events_df (pd.DataFrame): DataFrame z wydarzeniami
        forex_df_30min (pd.DataFrame): DataFrame z 30-minutowymi danymi forex
        forex_df_daily (pd.DataFrame): DataFrame z dziennymi danymi forex

    Returns:
        tuple: (model_short_term, model_daily) - wytrenowane modele
    """
    short_term_model = ShortTermForexPredictor(time_window=30)
    X_short, y_short = short_term_model.prepare_data(events_df, forex_df)

    input_dim = X_short.shape[1]
    output_dim = y_short.shape[1]
    short_term_model.model = short_term_model.create_base_model(input_dim, output_dim)
    short_term_history = short_term_model.train(X_short, y_short)

    # Model dzienny
    daily_model = DailyForexPredictor()
    X_daily, y_daily = daily_model.prepare_data(events_df, forex_df)

    input_dim = X_daily.shape[1]
    output_dim = y_daily.shape[1]
    daily_model.model = daily_model.create_base_model(input_dim, output_dim)
    daily_history = daily_model.train(X_daily, y_daily)

    return short_term_model, daily_model


# Przykład użycia:
if __name__ == "__main__":
    # Wczytanie danych
    events_df = pd.read_csv("events.csv")
    forex_df_30min = pd.read_csv("forex_30min.csv")
    forex_df_daily = pd.read_csv("forex_daily.csv")

    # Trenowanie modeli
    model_short_term, model_daily = train_models(
        events_df, forex_df_30min, forex_df_daily
    )
