from dataclasses import dataclass
import torch
import torch.distributed as dist
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh



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
        self.device_mesh = mesh_context.device_mesh
        self.pp_mesh = get_flat_mesh(self.device_mesh, "pp")
        self.draft_model = draft_model
        self._update_embedding = False
        self._update_lm_head = False
        self._update_embedding_adapters = False
        self._update_lm_head_adapters = False
        self.embedding = None
        self.lm_head = None
        self._discover_embedding()
        self._discover_lm_head()

    def _discover_embedding(self) -> None:
        """
        Check if any model part on the current rank 
        has an embedding layer.
        """
        for mp in self.model_parts:
            embedding = mp.get_input_embeddings()
            if embedding is not None:
                has_embedding = True
                self.embedding = embedding
                # Embedding layers should not have lora adapters
                # so we can just check if the embedding weight is trainable
                if embedding.weight.requires_grad:
                    self._update_embedding = True
                break
        dist.barrier()
        has_embed = torch.tensor([int(has_embedding)], 
                                 dtype=torch.int32, 
                                 device=self.device_mesh.device)
        pp_group = self.pp_mesh.get_group()
        dist.all_reduce(has_embed, op=dist.ReduceOp.SUM, group=pp_group)
        assert has_embed[0] == 1

    def _discover_lm_head(self) -> None:
        """
        Check if any model part on the current rank 
        has an lm_head layer.
        """
        for mp in self.model_parts:
            lm_head = mp.get_output_embeddings()
            if lm_head is not None:
                has_lm_head = True
                self.lm_head = lm_head
                if hasattr(lm_head, "lora_A"):
                    self._update_lm_head_adapters = True
                elif lm_head.weight.requires_grad:
                    self._update_lm_head = True
                break
        dist.barrier()
        has_lm_head = torch.tensor([int(has_lm_head)], 
                                 dtype=torch.int32, 
                                 device=self.device_mesh.device)
        pp_group = self.pp_mesh.get_group()
        dist.all_reduce(has_lm_head, op=dist.ReduceOp.SUM, group=pp_group)
        assert has_embed[0] == 1

    
    def capture(self):
        raise NotImplementedError

    def maybe_sync(self, step):
        raise NotImplementedError
