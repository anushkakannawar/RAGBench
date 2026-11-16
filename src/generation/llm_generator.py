"""
LLM Generator for RAG response generation.
Supports local models (via transformers) and API-based models.
"""
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LocalLLMGenerator:
    """
    Local LLM generator using Hugging Face transformers.
    Optimized for Colab T4 GPU with 4-bit quantization.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        max_new_tokens: int = 256
    ):
        """
        Initialize local LLM.
        
        Args:
            model_name: Model name from Hugging Face
            device: Device to load model on
            load_in_4bit: Use 4-bit quantization (saves memory)
            max_new_tokens: Max tokens to generate
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        print(f"[LocalLLM] Loading model: {model_name}")
        
        # Configure 4-bit quantization if enabled
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True
        )
        
        print(f"[LocalLLM] Model loaded successfully")
    
    def _format_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Format RAG prompt with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved chunks with 'text' key
            
        Returns:
            Formatted prompt string
        """
        # Combine context
        context_text = "\n\n".join([
            f"[{i+1}] {chunk['text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        context_chunks: List[Dict],
        temperature: float = 0.1,
        do_sample: bool = False
    ) -> str:
        """
        Generate answer given query and context.
        
        Args:
            query: User query
            context_chunks: Retrieved chunks
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer text
        """
        # Format prompt
        prompt = self._format_prompt(query, context_chunks)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = full_output[len(prompt):].strip()
        
        return answer
    
    def batch_generate(
        self,
        queries: List[str],
        context_chunks_batch: List[List[Dict]],
        temperature: float = 0.1,
        do_sample: bool = False
    ) -> List[str]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of queries
            context_chunks_batch: List of context chunk lists
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            List of generated answers
        """
        answers = []
        for query, chunks in zip(queries, context_chunks_batch):
            answer = self.generate(query, chunks, temperature, do_sample)
            answers.append(answer)
        
        return answers


class APILLMGenerator:
    """
    Generator using API-based LLMs (OpenAI, Groq, etc.).
    More cost-effective for experimentation.
    """
    
    def __init__(
        self,
        provider: str = "groq",
        model_name: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        max_tokens: int = 256
    ):
        """
        Initialize API-based LLM.
        
        Args:
            provider: 'openai' or 'groq'
            model_name: Model identifier
            api_key: API key (or set via environment variable)
            max_tokens: Max tokens to generate
        """
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        if provider == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=api_key)
                print(f"[APIGenerator] Using Groq: {model_name}")
            except ImportError:
                raise ImportError("Install groq: pip install groq")
        
        elif provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print(f"[APIGenerator] Using OpenAI: {model_name}")
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _format_messages(self, query: str, context_chunks: List[Dict]) -> List[Dict]:
        """Format as chat messages."""
        context_text = "\n\n".join([
            f"[{i+1}] {chunk['text']}" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        system_msg = "You are a helpful assistant. Answer questions based ONLY on the provided context."
        
        user_msg = f"""Context:
{context_text}

Question: {query}"""
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    
    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using API."""
        messages = self._format_messages(query, context_chunks)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def batch_generate(
        self,
        queries: List[str],
        context_chunks_batch: List[List[Dict]]
    ) -> List[str]:
        """Generate answers for multiple queries."""
        answers = []
        for query, chunks in zip(queries, context_chunks_batch):
            answer = self.generate(query, chunks)
            answers.append(answer)
        
        return answers


def create_generator(
    generator_type: str = "local",
    **kwargs
) -> LocalLLMGenerator | APILLMGenerator:
    """
    Factory function to create generator.
    
    Args:
        generator_type: 'local' or 'api'
        **kwargs: Arguments for specific generator
        
    Returns:
        Generator instance
    """
    if generator_type == "local":
        return LocalLLMGenerator(**kwargs)
    elif generator_type == "api":
        return APILLMGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


if __name__ == "__main__":
    # Example usage
    sample_chunks = [
        {'text': 'Retrieval-Augmented Generation combines retrieval with generation.'},
        {'text': 'RAG systems retrieve relevant documents before generating answers.'}
    ]
    
    print("Testing generator...")
    print("\nNote: This requires a model to be downloaded.")
    print("Uncomment below to test:\n")
    
    # generator = create_generator(
    #     generator_type="local",
    #     model_name="Qwen/Qwen2.5-7B-Instruct"
    # )
    # 
    # answer = generator.generate(
    #     "What is RAG?",
    #     sample_chunks
    # )
    # print(f"Answer: {answer}")
