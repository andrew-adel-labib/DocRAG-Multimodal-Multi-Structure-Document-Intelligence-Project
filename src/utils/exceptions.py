class RAGException(Exception):
    """Base exception for RAG system"""
    pass


class ParsingError(RAGException):
    pass


class ChunkingError(RAGException):
    pass


class EmbeddingError(RAGException):
    pass


class RetrievalError(RAGException):
    pass


class EvaluationError(RAGException):
    pass