class EwiserClassifier:
    def __init__(
        self,
        checkpoint: str,
        spacy_model: str,
        device: str,
        language: str,
        **disambiguator_kwargs
    ):
        self.checkpoint_path = checkpoint
        self.disambiguator_kwargs = disambiguator_kwargs
        self.language = language
        self.device = device
        self.spacy_model = spacy_model

        self.processor = None

    def import_load(self):
        import ewiser.fairseq_ext
        import nltk
        import spacy
        from ewiser.spacy.disambiguate import Disambiguator

        nltk.download("wordnet")  #  Needed inside EWISER
        classifier = Disambiguator(
            self.checkpoint_path,
            self.language,
            save_wsd_details=False,
            **self.disambiguator_kwargs
        ).eval()
        classifier.to(self.device)
        self.processor = spacy.load(self.spacy_model, disable=["parser", "ner"])
        classifier.enable(self.processor, "wsd")

    def is_loaded(self):
        return self.processor is not None
