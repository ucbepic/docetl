"""
Prompt Library for DocETL Operations and Rewrite Directives
"""

class PromptLibrary:
    
    @staticmethod
    def map_operator() -> str:
        prompt = """
        The map operator applies a semantic projection to each document in the dataset independently using the specified prompt. 
        """
        return prompt.strip()
    
    @staticmethod
    def reduce_operator() -> str:
        prompt = """
        The reduce operator aggregates information across multiple documents based on a set of user-specified keys, 
        ultimately producing one output document per unique combination of attribute values. 
        If the given group of documents is too large for the LLM to correctly process, we apply a batched folding 
        approach that starts with an empty accumulator and sequentially folds in batches of more than one document at a time. 
        """
        return prompt.strip()
    
    @staticmethod
    def resolve_operator() -> str:
        prompt = """
        This operator canonicalises one or more keys across documents that represent slight variations of the 
        same entity for subsequent grouping and aggregation. 
        """
        return prompt.strip()
    
    @staticmethod
    def gather_operator() -> str:
        prompt = """
        The gather operation complements the split operation by augmenting individual chunks with peripheral 
        information  necessary for understanding the chunk’s content.
        """
        return prompt.strip()
    

    # Below are the rewrite directives
    @staticmethod
    def document_chunking() -> str:
        prompt = """
        Use this directive when a map operation is applied to large documents that exceed LLM context windows or effective reasoning capabilities. 

        This directive rewrites a map operation into a pipeline: split -> gather -> map -> reduce. 

        Split: Split the document into smaller chunks creating as many new docs as there are chunks. 
        You can suggest a reasonable chunk size for this splitting step so that each chunk can fit comfortably 
        within the LLM’s context limit. 

        Gather: For each chunk, augment with relevant context from nearby chunks to help understand the chunk’s content. 
        You need to decide which surrounding chunks to include and how many. You have choices of inclusion of full chunks, 
        portions of chunks, or transformations (e.g.,  summaries) of chunks.

        Map: Apply a modified map operation per chunk.

        Reduce: Aggregate the individual chunk-level results into a single, coherent output that represents the analysis of 
        the entire original document.

        Example:
            Before:
                map: summarize the entire document
            After:
                split: split the document into contiguous chunks
                gather: gather k neighboring chunks before/after as context
                map: generate summarization per chunk
                reduce: global summarization across all chunks

        """
        return prompt.strip()
    
    @staticmethod
    def multi_level_agg() -> str:
        prompt = """
        Use this directive when a reduce operation aggregates a large set of documents at once, or if aggregations
        might lose nuance when performed in a single step, use this directive.
        This directive rewrites a reduce operation into a pipeline: fine-grained-reduce → roll-up-reduce.

        Fine-Grained Reduce: Perform aggregation within smaller, logical subgroups.
        Roll-Up Reduce: Next, aggregate these subgroup results into the final desired output.

        Example:
        Before:
            reduce: summarizing voting patterns by state
        After:
            reduce1: aggregate data by state and city
            reduce2: combine these city-level summaries to the state level
        """
        return prompt.strip()

    @staticmethod
    def dup_key_resolution() -> str:
        prompt = """
        Use this directive when a reduce operation needs to aggregate by keys that have ambiguous or semantically 
        equivalent values (e.g., "NYC", "New York City").

        This directive rewrites a reduce operation into a pipeline: resolve → reduce.

        resolve: use a resolve operation on each key independently to canonicalize duplicate values.
        reduce: perform the intended reduce operation on the cleaned, deduplicated keys. 
        """
        return prompt.strip()
    


if __name__ == "__main__":
    
    print("\nExample map prompt:")
    print(PromptLibrary.map_operator())

