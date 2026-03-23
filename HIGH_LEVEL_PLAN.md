1. Pretrain the task-VLM on symbolic plans and symbolic progress state.
2. Fine-tune the VLA on concrete grounded subatomic execution.
3. Use a grounding module between them that maps symbolic or semantic planner outputs to concrete scene IDs.
4. Run end-to-end RL with high-level rewards in symbolic space and low-level execution in concrete space.