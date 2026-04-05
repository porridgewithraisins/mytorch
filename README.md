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
