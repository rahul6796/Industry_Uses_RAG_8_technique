
    def _get_embedding(self, text:str)->List[float]:
        """
        Load the embedding model and return embeddings for a list of texts.

        Args:
            texts (list[str]): List of input strings.
            model_name (str): HuggingFace SentenceTransformer model name.

        Returns:
            numpy.ndarray: Embeddings matrix.
        """
        try:

            model = SentenceTransformer(APP_CONFIG.embedding_model)
            embeddings = model.encode(text)
            return embeddings
        except Exception as e:
            print(f'Error getting Embedding ! {e}')
            return []
