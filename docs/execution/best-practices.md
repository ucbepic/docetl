# Best Practices for DocETL

This guide outlines key best practices for using DocETL effectively, focusing on the most important aspects of pipeline creation, execution, and optimization.

## Pipeline Design

1. **Start Simple**: Begin with a basic pipeline and gradually add complexity as needed.
2. **Modular Design**: Break down complex tasks into smaller, manageable operations.
3. **Optimize Incrementally**: Optimize one operation at a time to ensure stability and verify improvements.

## Schema and Prompt Design

1. **Keep Schemas Simple**: Use simple output schemas whenever possible. Complex nested structures can be difficult for LLMs to produce consistently.
2. **Clear and Concise Prompts**: Write clear, concise prompts for LLM operations, providing relevant context from input data. Instruct quantities (e.g., 2-3 insights, one summary) to guide the LLM.
3. **Take advantage of Jinja Templating**: Use Jinja templating to dynamically generate prompts and provide context to the LLM. Feel free to use if statements, loops, and other Jinja features to customize prompts.
4. **Validate Outputs**: Use the `validate` field to ensure the quality and correctness of processed data. This consists of Python statements that validate the output and optionally retry the LLM if one or more statements fail.

## Handling Large Documents and Entity Resolution

1. **Chunk Large Inputs**: For documents exceeding token limits, consider using the optimizer to automatically chunk inputs.
2. **Use Resolve Operations**: Implement resolve operations before reduce operations when dealing with similar entities. Take care to write the compare prompts well to guide the LLM--often the optimizer-synthesized prompts are too generic.

## Optimization and Execution

1. **Use the Optimizer**: Leverage DocETL's optimizer for complex pipelines or when dealing with large documents.
2. **Leverage Caching**: Take advantage of DocETL's caching mechanism to avoid redundant computations.
3. **Monitor Resource Usage**: Keep an eye on API costs and processing time, especially when optimizing.
