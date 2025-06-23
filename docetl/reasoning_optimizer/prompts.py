"""
Prompt Library for DocETL Operations

This module contains a collection of prompt templates for various DocETL operations.
Each function returns a Jinja2 template string that can be used in DocETL configurations.
"""

from typing import Dict, Any, Optional


class PromptLibrary:
    """
    A library of prompt templates for DocETL operations.
    
    Each method returns a Jinja2 template string that can be used directly
    in DocETL operation configurations.
    """
    
    @staticmethod
    def map_operator() -> str:
        """
        Basic map operation prompt template.
        
        Returns:
            str: Jinja2 template for map operations
        """
        prompt = """
        Analyze the following document and extract key information:

        Document: {{ input.text }}

        Please provide a structured analysis of this document.
        """
        return prompt.strip()
    
    @staticmethod
    def filter_operator() -> str:
        """
        Basic filter operation prompt template.
        
        Returns:
            str: Jinja2 template for filter operations
        """
        prompt = """
        Determine if the following document meets the specified criteria:

        Document: {{ input.text }}

        Evaluate whether this document should be included based on the criteria.
        """
        return prompt.strip()
    
    @staticmethod
    def reduce_operator() -> str:
        """
        Basic reduce operation prompt template.
        
        Returns:
            str: Jinja2 template for reduce operations
        """
        prompt = """
        Combine and synthesize the following documents:

        {% for item in inputs %}
        Document {{ loop.index }}: {{ item.text }}
        {% endfor %}

        Provide a comprehensive summary that consolidates all the information.
        """
        return prompt.strip()
    
    @staticmethod
    def resolve_operator() -> str:
        """
        Basic resolve operation prompt template.
        
        Returns:
            str: Jinja2 template for resolve operations
        """
        prompt = """
        Compare the following two documents and determine if they refer to the same entity:

        Document 1: {{ input1.text }}
        Document 2: {{ input2.text }}

        Analyze the similarity and provide a resolution.
        """
        return prompt.strip()
    
    @staticmethod
    def sentiment_analysis() -> str:
        """
        Sentiment analysis prompt template.
        
        Returns:
            str: Jinja2 template for sentiment analysis
        """
        prompt = """
        Analyze the sentiment of the following text:

        Text: {{ input.text }}

        Determine the overall sentiment and provide a confidence score.
        """
        return prompt.strip()
    
    @staticmethod
    def entity_extraction() -> str:
        """
        Entity extraction prompt template.
        
        Returns:
            str: Jinja2 template for entity extraction
        """
        prompt = """
        Extract named entities from the following text:

        Text: {{ input.text }}

        Identify and extract all named entities including people, organizations, locations, and dates.
        """
        return prompt.strip()
    
    @staticmethod
    def summarization() -> str:
        """
        Text summarization prompt template.
        
        Returns:
            str: Jinja2 template for text summarization
        """
        prompt = """
        Create a concise summary of the following text:

        Text: {{ input.text }}

        Provide a clear and comprehensive summary that captures the main points.
        """
        return prompt.strip()
    
    @staticmethod
    def classification() -> str:
        """
        Text classification prompt template.
        
        Returns:
            str: Jinja2 template for text classification
        """
        prompt = """
        Classify the following text into the appropriate category:

        Text: {{ input.text }}

        Determine the most suitable category for this text.
        """
        return prompt.strip()
    
    @staticmethod
    def question_answering() -> str:
        """
        Question answering prompt template.
        
        Returns:
            str: Jinja2 template for question answering
        """
        prompt = """
        Answer the following question based on the provided context:

        Context: {{ input.context }}
        Question: {{ input.question }}

        Provide a clear and accurate answer based on the context.
        """
        return prompt.strip()
    
    @staticmethod
    def translation() -> str:
        """
        Translation prompt template.
        
        Returns:
            str: Jinja2 template for translation
        """
        prompt = """
        Translate the following text to the target language:

        Text: {{ input.text }}
        Target Language: {{ input.target_language }}

        Provide an accurate translation that maintains the original meaning.
        """
        return prompt.strip()
    
    @staticmethod
    def code_generation() -> str:
        """
        Code generation prompt template.
        
        Returns:
            str: Jinja2 template for code generation
        """
        prompt = """
        Generate code based on the following requirements:

        Requirements: {{ input.requirements }}
        Programming Language: {{ input.language }}

        Create clean, well-documented code that meets the specified requirements.
        """
        return prompt.strip()
    
    @staticmethod
    def data_extraction() -> str:
        """
        Data extraction prompt template.
        
        Returns:
            str: Jinja2 template for data extraction
        """
        prompt = """
        Extract structured data from the following text:

        Text: {{ input.text }}

        Identify and extract all relevant data points in a structured format.
        """
        return prompt.strip()
    
    @staticmethod
    def content_moderation() -> str:
        """
        Content moderation prompt template.
        
        Returns:
            str: Jinja2 template for content moderation
        """
        prompt = """
        Evaluate the following content for moderation:

        Content: {{ input.text }}

        Assess whether this content violates any community guidelines or policies.
        """
        return prompt.strip()
    
    @staticmethod
    def topic_modeling() -> str:
        """
        Topic modeling prompt template.
        
        Returns:
            str: Jinja2 template for topic modeling
        """
        prompt = """
        Identify the main topics in the following text:

        Text: {{ input.text }}

        Extract and categorize the primary topics discussed in this content.
        """
        return prompt.strip()
    
    @staticmethod
    def custom_prompt(template: str) -> str:
        """
        Create a custom prompt from a template string.
        
        Args:
            template (str): The custom template string
            
        Returns:
            str: The custom prompt template
        """
        return template.strip()
    
    @staticmethod
    def get_prompt_by_name(name: str) -> Optional[str]:
        """
        Get a prompt by its name.
        
        Args:
            name (str): The name of the prompt to retrieve
            
        Returns:
            Optional[str]: The prompt template if found, None otherwise
        """
        prompt_methods = {
            'map': PromptLibrary.map_operator,
            'filter': PromptLibrary.filter_operator,
            'reduce': PromptLibrary.reduce_operator,
            'resolve': PromptLibrary.resolve_operator,
            'sentiment': PromptLibrary.sentiment_analysis,
            'entity': PromptLibrary.entity_extraction,
            'summary': PromptLibrary.summarization,
            'classify': PromptLibrary.classification,
            'qa': PromptLibrary.question_answering,
            'translate': PromptLibrary.translation,
            'code': PromptLibrary.code_generation,
            'extract': PromptLibrary.data_extraction,
            'moderate': PromptLibrary.content_moderation,
            'topic': PromptLibrary.topic_modeling,
        }
        
        method = prompt_methods.get(name.lower())
        return method() if method else None
    
    @staticmethod
    def list_available_prompts() -> list:
        """
        Get a list of all available prompt names.
        
        Returns:
            list: List of available prompt names
        """
        return [
            'map',
            'filter', 
            'reduce',
            'resolve',
            'sentiment',
            'entity',
            'summary',
            'classify',
            'qa',
            'translate',
            'code',
            'extract',
            'moderate',
            'topic'
        ]


# Convenience functions for easy access
def get_prompt(name: str) -> Optional[str]:
    """
    Get a prompt template by name.
    
    Args:
        name (str): The name of the prompt
        
    Returns:
        Optional[str]: The prompt template if found
    """
    return PromptLibrary.get_prompt_by_name(name)


def list_prompts() -> list:
    """
    Get a list of all available prompts.
    
    Returns:
        list: List of available prompt names
    """
    return PromptLibrary.list_available_prompts()


# Example usage:
if __name__ == "__main__":
    # Print all available prompts
    print("Available prompts:")
    for prompt_name in list_prompts():
        print(f"  - {prompt_name}")
    
    print("\nExample map prompt:")
    print(PromptLibrary.map_operator())
    
    print("\nExample sentiment analysis prompt:")
    print(PromptLibrary.sentiment_analysis())

