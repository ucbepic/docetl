# Community

Welcome to the DocETL community! We're excited to have you join us in exploring and improving document extraction and transformation workflows.

## Connect with Us

- **GitHub Repository**: Contribute to the project or report issues on our [GitHub repo](https://github.com/shreyashankar/docetl).
- **Discord Community**: Join our [Discord server](https://discord.gg/docetl) to chat with other users, ask questions, and share your experiences.
- **Lab Webpage**: Visit our [Lab Page](https://epic.berkeley.edu) for a description of our research.

## Roadmap and Ongoing Projects

We're constantly working to improve DocETL and explore new possibilities in document processing. Our current ideas span both research and engineering problems, and are organized into the following categories:

### User Interface and Interaction

1. **Interactive Pipeline Creation**: Developing intuitive interfaces for creating and optimizing DocETL pipelines interactively.
2. **Natural Language to DocETL Pipeline**: Building tools to generate DocETL pipelines from natural language descriptions.

### Debugging and Optimization

3. **DocETL Debugger**: Creating a debugger with provenance tracking, allowing users to visualize all intermediates that contributed to a specific output.
4. **Smarter Agent and Planning Architectures**: Optimizing plan exploration based on data characteristics. For instance, refining the optimizer to avoid unnecessary exploration of plans with the [gather operator](operators/gather.md) for tasks that don't require peripheral context when decomposing map operations for large documents.
5. **Plan Efficiency Optimization**: Implementing strategies (and devising new strategies) to reduce latency and cost for the most accurate plans. This includes batching LLM calls, using model cascades, and implementing parallel processing for independent operations.

### Data Handling and Storage

6. **Comprehensive Data Loading**: Expanding support beyond JSON to include formats like CSV and Apache Arrow, as well as loading from the cloud.
7. **New Storage Format**: Exploring a specialized storage format for unstructured data and documents, particularly suited for pipeline intermediates. For example, tokens that do not contribute much to the final output can be compressed further.

### Model and Tool Integration

8. **Model Diversity**: Extending support beyond OpenAI to include a wider range of models, with a focus on local models.
9. **OCR and PDF Extraction**: Improving integration with OCR technologies and PDF extraction tools for more robust document processing.

We welcome community contributions and ideas for these projects. If you're interested in contributing or have suggestions for new areas of exploration, please reach out on our Discord server or GitHub repository.

## Frequently Encountered Issues

### KeyError in Operations

If you're encountering a KeyError, it's often due to missing an unnest operation in your workflow. The unnest operation is crucial for flattening nested data structures.

**Solution**: Add an [unnest operation](operators/unnest.md) to your pipeline before accessing nested keys.
