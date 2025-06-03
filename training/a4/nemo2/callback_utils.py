import os

try:
    # TODO: Remove this once we have full transition to Lightning 2.0
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

import torch


class MemoryProfileCallback(pl.Callback):
    def __init__(
        self,
        file_prefix="memdump", 
        max_entries=1000000,
        rank_0_only=True,

        start_location="init",
        end_location="train_start",
        force_oom_before_stop=False,
    ):
        self.file_prefix = file_prefix
        self.max_entries = max_entries
        self.force_oom_before_stop = force_oom_before_stop

        # process group not initialized at this part, using this method to get the global rank and world size
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        # customize this list if other ranks need profiling
        profile_ranks = [0] if rank_0_only else list(range(world_size))
        self.do_profile = global_rank in profile_ranks

        self.start_location = start_location
        self.end_location = end_location

        self.maybe_start_here("init")

    def force_oom(self):
        # adds a bunch of large tensors here to trigger CUDA OOM
        print("[Memory Profiler] Forcing OOM")
        mems = []
        size = int(1e9)
        for idx in range(10):
            mem = torch.arange(start=size*idx, end=size*(idx+1), device=torch.device("cuda:0"), dtype=torch.int64) 
            mems.append(mem)
            total_mem_use = sum([m.nelement() * m.element_size() / 1000**3 for m in mems])
            print(f"[Memory Profiler] Total mem use: {total_mem_use} GB")
        print("[Memory Profiler]", sum([x[-5:].to(torch.device("cpu")) for x in mems]).tolist())

    def maybe_start_here(self, current_location):
        if self.do_profile and self.start_location == current_location:
            torch.cuda.memory._record_memory_history(max_entries = self.max_entries)

    def maybe_stop_here(self, current_location):
        if self.do_profile and self.end_location == current_location:
            # first trigger OOM before saving
            if self.force_oom_before_stop:
                self.force_oom()

            # saves the profiles
            filename = f"/mem_dump/{self.file_prefix}.pickle"
            torch.cuda.memory._dump_snapshot(filename)
            torch.cuda.memory._record_memory_history(enabled=None)

    def cleanup_after_oom(self):
        import traceback
        memdump_filename = f"/mem_dump/{self.file_prefix}_failed.pickle"
        traceback_filename = f"/mem_dump/{self.file_prefix}_failed.traceback"
        with open(traceback_filename, "w") as f:
            traceback.print_exc(file=f)
        torch.cuda.memory._dump_snapshot(memdump_filename)
        torch.cuda.memory._record_memory_history(enabled=None)
            
    def on_train_start(self, trainer, pl_module):
        self.maybe_start_here("train_start")
        self.maybe_stop_here("train_start")