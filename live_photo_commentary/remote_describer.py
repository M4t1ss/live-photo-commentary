"""Base class for remote API-based describers."""

from abc import abstractmethod

from .describer import (
    Describer,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_ENDING,
    DEFAULT_FIRST_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_HISTORY_PROMPT,
    DEFAULT_COMPACT_PROMPT,
)


class RemoteDescriber(Describer):
    """Abstract base class for remote API-based vision models with factory pattern."""
    
    def __new__(cls, provider="gemini", **kwargs):
        """Factory method that returns the appropriate subclass based on provider."""
        if cls is not RemoteDescriber:
            # Direct instantiation of subclass
            return super().__new__(cls)
        
        # Factory logic for RemoteDescriber instantiation with lazy imports
        if provider == "gemini":
            from .remote_describers.gemini import GeminiDescriber
            instance = super(RemoteDescriber, GeminiDescriber).__new__(GeminiDescriber)
        elif provider == "openai":
            from .remote_describers.openai import OpenAIDescriber
            instance = super(RemoteDescriber, OpenAIDescriber).__new__(OpenAIDescriber)
        else:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: gemini, openai"
            )
        
        return instance

    def __init__(self,
                 api_key=None,
                 system_prompt=DEFAULT_SYSTEM_PROMPT,
                 ending=DEFAULT_ENDING,
                 first_prompt=DEFAULT_FIRST_PROMPT,
                 prompt=DEFAULT_PROMPT,
                 history_prompt=DEFAULT_HISTORY_PROMPT,
                 compact_prompt=DEFAULT_COMPACT_PROMPT,
                 max_history_size=False,
                 min_history_size=False,
                 model_id=None,
                 provider="gemini",
                 **kwargs
    ):
        super().__init__(
            system_prompt=system_prompt,
            ending=ending,
            first_prompt=first_prompt,
            prompt=prompt,
            history_prompt=history_prompt,
            compact_prompt=compact_prompt,
            max_history_size=max_history_size,
            min_history_size=min_history_size,
        )
        
        self.provider = provider
        self.model_id = model_id or self.get_default_model()
        self.api_key = api_key
        
        self._setup_client()
    
    @abstractmethod
    def get_default_model(self):
        """Return the default model ID for this provider."""
        pass
    
    @abstractmethod
    def _setup_client(self):
        """Setup the API client."""
        pass
