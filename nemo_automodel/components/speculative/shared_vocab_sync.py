from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
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
        has_embedding = False
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
        assert has_embed[0] == 1, f"Expected only one rank to have an embedding layer per pipeline parallel process group, but found {has_embed[0]} such ranks."

        global_rank = dist.get_rank()
        src_tensor = torch.tensor(
            [global_rank if has_embedding else -1], 
            dtype=torch.long, 
            device="cuda"
        )
        dist.all_reduce(src_tensor, op=dist.ReduceOp.MAX, group=pp_group)
        self.embedding_src_rank = src_tensor.item()
        if self.embedding_src_rank == -1:
            raise RuntimeError("No sender rank identified for embedding layer.")

    def _discover_lm_head(self) -> None:
        """
        Check if any model part on the current rank 
        has an lm_head layer.
        """
        has_lm_head = False
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
        assert has_lm_head[0] == 1, "Expected only one rank to have an lm_head layer per pipeline parallel process group, but found {has_lm_head[0]} such ranks."
        global_rank = dist.get_rank()
        src_tensor = torch.tensor(
            [global_rank if has_lm_head else -1], 
            dtype=torch.long, 
            device="cuda"
        )
        dist.all_reduce(src_tensor, op=dist.ReduceOp.MAX, group=pp_group)
        self.lm_head_src_rank = src_tensor.item()
        if self.lm_head_src_rank == -1:
            raise RuntimeError("No sender rank identified for lm_head layer.")


    def capture(self):
        """
        Copy embeddings, lm_head and lora adapters from the target to the draft model.
        """
        # Embeddings
        embed_tensor = self.embedding.weight.full_tensor() if self.embedding is not None else torch.zeros_like(self.draft_model.embed_tokens.weight)
        dist.broadcast(embed_tensor, src=self.embedding_src_rank, group=self.pp_mesh.get_group())
        _write_full_into_param(self.draft_model.embed_tokens.weight, embed_tensor)
        del embed_tensor

        # lm head
        lm_head_tensor = self.lm_head.weight.full_tensor() if self.lm_head is not None else torch.zeros_like(self.draft_model.lm_head.weight)
        dist.broadcast(lm_head_tensor, src=self.lm_head_src_rank, group=self.pp_mesh.get_group())
        _write_full_into_param(self.draft_model.lm_head.weight, lm_head_tensor)
        del lm_head_tensor

        # Lora adapters
        if self._update_lm_head_adapters:
            lm_head_lora_A_tensor = self.lm_head.lora_A.weight.full_tensor() if self.lm_head is not None else torch.zeros_like(self.draft_model.lm_head.lora_A.weight)
            dist.broadcast(lm_head_lora_A_tensor, src=self.lm_head_src_rank, group=self.pp_mesh.get_group())
            _write_full_into_param(self.draft_model.lm_head.lora_A.weight, lm_head_lora_A_tensor)
            del lm_head_lora_A_tensor
            
            lm_head_lora_B_tensor = self.lm_head.lora_B.weight.full_tensor() if self.lm_head is not None else torch.zeros_like(self.draft_model.lm_head.lora_B.weight)
            dist.broadcast(lm_head_lora_B_tensor, src=self.lm_head_src_rank, group=self.pp_mesh.get_group())
            _write_full_into_param(self.draft_model.lm_head.lora_B.weight, lm_head_lora_B_tensor)
            del lm_head_lora_B_tensor

            lm_head_dora_magnitude = self.lm_head.dora_magnitude.weight if self.lm_head is not None else torch.zeros_like(self.draft_model.lm_head.dora_magnitude.weight)
            dist.broadcast(lm_head_dora_magnitude, src=self.lm_head_src_rank, group=self.pp_mesh.get_group())
            _write_full_into_param(self.draft_model.lm_head.dora_magnitude.weight, lm_head_dora_magnitude)
            del lm_head_dora_magnitude

    def maybe_sync(self, step):
        raise NotImplementedError

def _write_full_into_param(param: torch.nn.Parameter, full: torch.Tensor) -> None:
    """Write a full tensor (identical on every rank) into a possibly-sharded param."""
    with torch.no_grad():
        if not isinstance(param, DTensor):
            param.copy_(full)                     # ignored_params / unsharded case
            return
        # Read layout off the target — works for dp, or 2D HSDP placements, unchanged.
        src = distribute_tensor(full.to(param.dtype), param.device_mesh, param.placements)
        assert src.to_local().shape == param.to_local().shape, (
                f"shard mismatch: {src.to_local().shape} vs {param.to_local().shape}"
            )
            param.copy_(src)                          # local-to-local; param object preserved


    
