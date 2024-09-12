# Vision

Things I'd like for the interface/agents to do:

- Don't synthesize operators that are too complex. No complex schemas. Intermediate steps should be as unstructured as possible, leaving structured generation to the final output.
- Optimize _one operation_ at a time. This way the user can see the intermediate steps and understand what's happening, and help modify prompts as needed.
- Users should be in control of validation prompts.
- When users are looking at intermediates, we should have the ability to run validators on the intermediate prompts.
- We need to store intermediates and have provenance.
- Have an interface to interactively create docetl pipelines. Start by users defining a high-level task, and optimize one operation at a time.
