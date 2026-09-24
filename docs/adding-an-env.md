# Adding an environment

The contributor guide is now split by responsibility:

- [Choose an environment, task, or renderer change](adding/index.md).
- [Add an environment backend and exhaustive State](adding/environment.md).
- [Add a task or registration on an existing backend](adding/task.md).
- [Add a renderer and prove its fidelity](adding/renderer.md).

This URL remains for existing links. The former renderer-first PushT walkthrough
is superseded: compact RenderState is not an exhaustive simulator checkpoint,
and PushT's historical semantic recovery remains incomplete because contact
history could not be safely restored. Use accepted implementations in the new
guides as starting examples.
