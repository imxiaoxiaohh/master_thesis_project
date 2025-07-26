"""Unified LLM client for different providers."""

def create_client(config, api_key):
    """Create appropriate LLM client based on provider."""
    
    if config.provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        def generate(prompt):
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2
            )
            return response.choices[0].message.content
    
    elif config.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        def generate(prompt):
            response = client.messages.create(
                model=config.model,
                max_tokens=1024,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    elif config.provider == "deepseek":
        import openai
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        def generate(prompt):
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                # max_tokens=1024,
                temperature=0.2,
                stream=False
            )
            return response.choices[0].message.content
    
    elif config.provider == "together":
        from together import Together
        client = Together()
        
        def generate(prompt):
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role":"system", "content":"You are a helpful mathematical assistant. Do **not** output any <think> or </think> tags or any internal reasoning. Only emit the final answer "},
                    {"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2,
                
            )
            return response.choices[0].message.content
    
    elif config.provider == "huggingface":
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=api_key)
        
        def generate(prompt):
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2
            )
            return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unknown provider: {config.provider}")
    
    return generate