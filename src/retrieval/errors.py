
class IndexNotFoundError(Exception):
    """Errore sollevato quando un indice richiesto non esiste in Pinecone."""


class EmbeddingDimensionMismatchError(Exception):
    """Errore sollevato quando le dimensioni dell'embedding non coincidono con quelle dell'indice."""