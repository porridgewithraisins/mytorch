# mytorch

In-progress implementation of a full tensor frontend and backend, for learning purposes.

## Done

- Eagerly executing frontend with a numpy CPU backend.
- Eager autograd support
- Frontend operator fusion
- grad(grad) support

## In progress

- Lazy tensor backend, outputting an AST
- AST level optimisations
- Backend operator fusions

## To-do

- Shape inference IR
- Flash attention CUDA implementation. Wire up as a backend fusion operator
- MLIR/Triton/TensorRT-LLM/numpy backend for the other operators

## Status / Goals

This is primarily a showcase project - the goal is to demonstrate the idea end to end, not replace PyTorch.

Concretely, two fusions will be fully implemented:

- *Frontend fusion: log-softmax* - pattern matched on the DAG, fused into a single op before lowering
- *Backend fusion: flash attention* - hand-written CUDA kernel, automatically dispatched when the attention subgraph is detected

The flash attention case is interesting beyond just attention - the underlying pattern is
*contraction → pointwise → contraction* where the intermediate tensor is too large to materialize.
In principle the same tiling strategy generalizes to any op matching that shape.
Flash attention is just the most important instance of it.
