from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from collections.abc import Callable



@dataclass
class SharedVocabSyncConfig:
    sync_interval: int = 10

    def build(self, *, model_parts, mesh_context, draft_model, lora_func) -> "SharedVocabSync":
        sync = SharedVocabSync(self, model_parts, mesh_context, draft_model)
        sync.capture(copy_embedding=True, copy_lm_head=True, lora_func=lora_func)
        return sync

class SharedVocabSync:
    """
    Synchronizes embedding and lm_head, and their associated lora adapters, 
    between target and draft models for concurrent training.

    The following scenarios are currently unsupported:
    - qLoRA (this falls back to the transformers implementation)
    - tied weights for embeddings and lm_head 
    - tensor parallelism for weights and lm_head
    """
    def __init__(self, cfg, model_parts, mesh_context, draft_model):
        self.cfg = cfg
        self.model_parts = model_parts
        self.mesh_context = mesh_context
        self.device_mesh = mesh_context.device_mesh
        pp_mesh = get_flat_mesh(self.device_mesh, "pp")
        self.pp_group = pp_mesh.get_group()
        self.pp_degree = mesh_context.pp_size
        self.draft_model = draft_model
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
        update_embedding = False
        for mp in self.model_parts:
            embedding = mp.get_input_embeddings()
            if embedding is not None:
                has_embedding = True
                self.embedding = embedding
                if embedding.weight.requires_grad:
                    update_embedding = True
                break
        dist.barrier()
        embed_count = torch.tensor([int(has_embedding)], 
                                 dtype=torch.long, 
                                 device="cuda")
        if self.pp_degree > 1:
            dist.all_reduce(embed_count, op=dist.ReduceOp.SUM, group=self.pp_group)
        assert embed_count.item() == 1, f"Expected only one rank to have an embedding layer per pipeline parallel process group, but found {has_embed[0]} such ranks."

        update_embed = torch.tensor([int(update_embedding)], 
                                 dtype=torch.long, 
                                 device="cuda")
        if self.pp_degree > 1:
            dist.all_reduce(update_embed, op=dist.ReduceOp.MAX, group=self.pp_group)
        self.update_embedding = bool(update_embed.item())

        global_rank = dist.get_rank()
        src_tensor = torch.tensor(
            [global_rank if has_embedding else -1], 
            dtype=torch.long, 
            device="cuda"
        )
        if self.pp_degree > 1:
            dist.all_reduce(src_tensor, op=dist.ReduceOp.MAX, group=self.pp_group)
        self.embedding_src_rank = src_tensor.item()
        if self.embedding_src_rank == -1:
            raise RuntimeError("No sender rank identified for embedding layer.")

    def _discover_lm_head(self) -> None:
        """
        Check if any model part on the current rank 
        has an lm_head layer.
        """
        has_lm_head = False
        update_lm_head_adapters = False
        update_lm_head = False
        update_lm_head_dora = False
        for mp in self.model_parts:
            lm_head = mp.get_output_embeddings()
            if lm_head is not None:
                has_lm_head = True
                self.lm_head = lm_head
                if hasattr(lm_head, "lora_A"):
                    update_lm_head_adapters = True
                    update_lm_head_dora = getattr(lm_head, "use_dora", False)
                    update_lm_head = False
                elif lm_head.weight.requires_grad:
                    update_lm_head_adapters = False
                    update_lm_head_dora = False
                    update_lm_head = True
                break
        dist.barrier()
        head_count = torch.tensor([int(has_lm_head)], 
                                 dtype=torch.long, 
                                 device="cuda")
        if self.pp_degree > 1:
            dist.all_reduce(head_count, op=dist.ReduceOp.SUM, group=self.pp_group)
        assert head_count.item() == 1, f"Expected only one rank to have an lm_head layer per pipeline parallel process group, but found {has_lm_head[0]} such ranks."


        update_lm_head = torch.tensor([int(update_lm_head)], 
                                 dtype=torch.long, 
                                 device="cuda")
        if self.pp_degree > 1:
            dist.all_reduce(update_lm_head, op=dist.ReduceOp.MAX, group=self.pp_group)
        self.update_lm_head = bool(update_lm_head.item())

        update_lm_head_adapters = torch.tensor([int(update_lm_head_adapters)], 
                                 dtype=torch.long, 
                                 device="cuda")
        if self.pp_degree > 1:
            dist.all_reduce(update_lm_head_adapters, op=dist.ReduceOp.MAX, group=self.pp_group)
        self.update_lm_head_adapters = bool(update_lm_head_adapters.item())

        update_lm_head_dora = torch.tensor([int(update_lm_head_dora)], 
                                 dtype=torch.long, 
                                 device="cuda")
        if self.pp_degree > 1:
            dist.all_reduce(update_lm_head_dora, op=dist.ReduceOp.MAX, group=self.pp_group)
        self.update_lm_head_dora = bool(update_lm_head_dora.item())

        global_rank = dist.get_rank()
        src_tensor = torch.tensor(
            [global_rank if has_lm_head else -1], 
            dtype=torch.long, 
            device="cuda"
        )
        if self.pp_degree > 1:
            dist.all_reduce(src_tensor, op=dist.ReduceOp.MAX, group=self.pp_group)
        self.lm_head_src_rank = src_tensor.item()
        if self.lm_head_src_rank == -1:
            raise RuntimeError("No sender rank identified for lm_head layer.")


    def capture(self, copy_embedding:bool=False, copy_lm_head:bool=False, lora_func:Callable=None):
        """
        Copy embeddings, lm_head and lora adapters from the target to the draft model.
        Apply lora_func to the lm_head if provided. This is a one-time operation performed 
        during setup only.
        """

        if lora_func is not None:
            self.draft_model.apply_lora_lm_head(lora_func)

        # Embeddings
        draft_embed = self.draft_model.embed_tokens.weight
        if copy_embedding:
            embed_tensor = _gather_full(self.embedding).to(draft_embed.dtype) if self.embedding is not None else torch.empty(draft_embed.shape, dtype=draft_embed.dtype, device=draft_embed.device)
            if self.pp_degree > 1:
                dist.broadcast(embed_tensor, src=self.embedding_src_rank, group=self.pp_group)
            _write_full_into_param(draft_embed, embed_tensor)
            del embed_tensor

        # lm head
        draft_lm_head = self.draft_model.lm_head.weight
        if copy_lm_head:
            lm_head_tensor = _gather_full(self.lm_head).to(draft_lm_head.dtype) if self.lm_head is not None else torch.empty(draft_lm_head.shape, dtype=draft_lm_head.dtype, device=draft_lm_head.device)
            if self.pp_degree > 1:
                dist.broadcast(lm_head_tensor, src=self.lm_head_src_rank, group=self.pp_group)
            _write_full_into_param(draft_lm_head, lm_head_tensor)
            del lm_head_tensor

        # Lora adapters
        if self.update_lm_head_adapters:
            draft_lm_head_lora_A = self.draft_model.lm_head.lora_A.weight
            lm_head_lora_A_tensor = _gather_full(self.lm_head.lora_A).to(draft_lm_head_lora_A.dtype) if self.lm_head is not None else torch.empty(draft_lm_head_lora_A.shape, dtype=draft_lm_head_lora_A.dtype, device=draft_lm_head_lora_A.device)
            if self.pp_degree > 1:
                dist.broadcast(lm_head_lora_A_tensor, src=self.lm_head_src_rank, group=self.pp_group)
            _write_full_into_param(draft_lm_head_lora_A, lm_head_lora_A_tensor)
            del lm_head_lora_A_tensor
            
            draft_lm_head_lora_B = self.draft_model.lm_head.lora_B.weight
            lm_head_lora_B_tensor = _gather_full(self.lm_head.lora_B).to(draft_lm_head_lora_B.dtype) if self.lm_head is not None else torch.empty(draft_lm_head_lora_B.shape, dtype=draft_lm_head_lora_B.dtype, device=draft_lm_head_lora_B.device)
            if self.pp_degree > 1:
                dist.broadcast(lm_head_lora_B_tensor, src=self.lm_head_src_rank, group=self.pp_group)
            _write_full_into_param(self.draft_model.lm_head.lora_B.weight, lm_head_lora_B_tensor)
            del lm_head_lora_B_tensor

        if self.update_lm_head_dora:
            draft_lm_head_lora_magnitude = self.draft_model.lm_head.lora_magnitude
            lm_head_lora_magnitude = _gather_full(self.lm_head.lora_magnitude) if self.lm_head is not None else torch.empty(draft_lm_head_lora_magnitude.shape, dtype=draft_lm_head_lora_magnitude.dtype, device=draft_lm_head_lora_magnitude.device)
            if self.pp_degree > 1:
                dist.broadcast(lm_head_lora_magnitude, src=self.lm_head_src_rank, group=self.pp_group)
            _write_full_into_param(draft_lm_head_lora_magnitude, lm_head_lora_magnitude)
            del lm_head_lora_magnitude
        self.draft_model.set_embedding_head_trainable(False)

    def update_draft_embed_lm_head(self, step:int) -> None:
        if step % self.cfg.sync_interval != 0:
            return
        self.capture(copy_embedding=self.update_embedding, copy_lm_head=self.update_lm_head)

def _gather_full(t: torch.Tensor) -> torch.Tensor:
    """Full, plain, detached tensor from a possibly-sharded param."""
    return (t.full_tensor() if hasattr(t, "full_tensor") else t).detach()

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


    
