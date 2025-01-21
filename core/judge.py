from typing import List, Tuple

import numpy as np
import pandas as pd
from fmp.consts import TradingAction
from fmp.repository.models import TradingRecommendation
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model

from core.ai import DailyForexPredictor, ShortTermForexPredictor


class ForexTradingSystem:
    def __init__(
        self,
        short_term_model: ShortTermForexPredictor,
        daily_model: DailyForexPredictor,
        confidence_threshold: float = 0.6,
        min_price_change: float = 0.001,
    ):
        self.short_term_model = short_term_model
        self.daily_model = daily_model
        self.confidence_threshold = confidence_threshold
        self.min_price_change = min_price_change

    def _calculate_confidence(
        self, short_term_pred: np.ndarray, daily_pred: np.ndarray
    ) -> Tuple[float, TradingAction]:
        """
        Oblicza pewność predykcji i sugerowaną akcję na podstawie przewidywań obu modeli.
        """
        # Średnia ważona predykcji (większa waga dla modelu krótkoterminowego)
        short_term_weight = 0.7
        daily_weight = 0.3

        # Bierzemy pod uwagę głównie zmiany close
        short_term_change = short_term_pred[3]  # indeks 3 to close_change
        daily_change = daily_pred[3]

        weighted_change = (
            short_term_change * short_term_weight + daily_change * daily_weight
        )

        confidence = abs(weighted_change)

        if confidence < self.min_price_change:
            action = TradingAction.HOLD
        elif weighted_change > 0:
            action = TradingAction.BUY
        else:
            action = TradingAction.SELL

        return confidence, action

    def get_recommendations(
        self, current_events: pd.DataFrame, current_forex_data: pd.DataFrame
    ) -> List[TradingRecommendation]:
        """
        Generuje rekomendacje handlowe na podstawie bieżących wydarzeń i danych forex.
        """
        recommendations = []

        # Grupujemy wydarzenia według waluty
        for currency, currency_events in current_events.groupby("currency"):
            # Znajdujemy odpowiednie pary walutowe
            currency_pairs = current_forex_data[
                current_forex_data["base_currency"] == currency
            ]

            for _, pair_data in currency_pairs.groupby("ticker"):
                # Przygotowujemy dane dla obu modeli
                X_short = self.short_term_model.preprocess_events(currency_events)
                X_daily = self.daily_model.preprocess_events(currency_events)

                # Otrzymujemy predykcje
                short_term_pred = self.short_term_model.model.predict(X_short).mean(
                    axis=0
                )
                daily_pred = self.daily_model.model.predict(X_daily).mean(axis=0)

                # Obliczamy pewność i akcję
                confidence, action = self._calculate_confidence(
                    short_term_pred, daily_pred
                )

                # Jeśli pewność jest wystarczająca, dodajemy rekomendację
                if (
                    confidence >= self.confidence_threshold
                    or action == TradingAction.HOLD
                ):
                    short_term_dict = {
                        "open_change": float(short_term_pred[0]),
                        "high_change": float(short_term_pred[1]),
                        "low_change": float(short_term_pred[2]),
                        "close_change": float(short_term_pred[3]),
                    }

                    daily_dict = {
                        "open_change": float(daily_pred[0]),
                        "high_change": float(daily_pred[1]),
                        "low_change": float(daily_pred[2]),
                        "close_change": float(daily_pred[3]),
                    }

                    recommendations.append(
                        TradingRecommendation(
                            pair=pair_data["ticker"].iloc[0],
                            action=action,
                            confidence=float(confidence),
                            short_term_prediction=short_term_dict,
                            daily_prediction=daily_dict,
                            supporting_events=currency_events.to_dict("records"),
                        )
                    )

        # Sortujemy rekomendacje według pewności
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations

    @classmethod
    def make_decisions(
        cls,
        events_df: pd.DataFrame,
        forex_data: pd.DataFrame,
        short_term_model: Model,
        daily_model: Model,
    ) -> List[TradingRecommendation]:
        trading_system = cls(
            short_term_model=short_term_model,
            daily_model=daily_model,
            confidence_threshold=0.6,
            min_price_change=0.001,
        )

        return trading_system.get_recommendations(
            current_events=events_df, current_forex_data=forex_data
        )

    @staticmethod
    def load_model(model_path: str) -> Model:
        return load_model(model_path)
