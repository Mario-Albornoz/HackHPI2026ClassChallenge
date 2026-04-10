from handlers.FeatureEngineerer import FeatureEngineerer
from handlers.Preprocessor import Preprocessor
from handlers.Transformer import Transformer
from handlers.Visualizator import Visualizator


class HandlerOrchestrator:
    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self.transformer = Transformer()
        self.visualizator = Visualizator()
        self.feature_engineerer = FeatureEngineerer()

        pass
    
    def preproccess_data(self):
        self.preprocessor.clean_all(                                                                                                                   
            annotations_root="data/annotation",                                                                                                   
            images_root="data/data",                                                                                                               
            output_root="data/annotations_clean",                                                                                                         
        )

        return self