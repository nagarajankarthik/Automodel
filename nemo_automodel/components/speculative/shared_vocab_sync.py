from dataclasses import dataclass



@dataclass
class SharedVocabSyncConfig:
    sync_every_steps: int = 100

    def build(self, *, model_parts, mesh_context, draft_model) -> "SharedVocabSync":
        sync = SharedVocabSync(self, model_parts, mesh_context, draft_model)
        sync.capture()   # probe ownership, derive per-module sync policy
        return sync

class SharedVocabSync:
    """
    Synchronizes embedding and lm_head, and their associated lora adapters, 
    between target and draft models for concurrent training.
    """

    def __init__(self, cfg, model_parts, mesh_context, draft_model):
        self.cfg = cfg
        self.model_parts = model_parts
        self.mesh_context = mesh_context
        self.draft_model = draft_model
        self._has_embedding = self._holds_embedding()
        self._has_lm_head = self._holds_lm_head()

    def _holds_embedding(self):
        """
        Check if any model part on the current rank 
        has an embedding layer.
        """
        for mp in self.model_parts:
            embedding = mp.get_input_embeddings()
            if embedding is not None:
                return True
        return False

    def _holds_lm_head(self):
        """
        Check if any model part on the current rank 
        has an lm_head layer.
        """
        for mp in self.model_parts:
            lm_head = mp.get_output_embeddings()
            if lm_head is not None:
                return True
        return False

    def capture(self):
        raise NotImplementedError

    def maybe_sync(self, step):
        raise NotImplementedError
