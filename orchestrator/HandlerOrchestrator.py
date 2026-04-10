from handlers.FeatureEngineerer import FeatureEngineerer
from handlers.Preprocessor import Preprocessor
from handlers.Transformer import Transformer
from handlers.Visualizator import Visualizator


class HandlerOrchestrator:
    def __init__(self) -> None:
        self.prepocessor = Preprocessor()
        self.transformer = Transformer()
        self.visualizator = Visualizator()
        self.feature_engineerer = FeatureEngineerer()

        pass
