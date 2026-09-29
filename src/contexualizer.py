

def build_model_context(prompt: str, registry: list[dict]) -> str:
    context = "Available functions:\n"

    for func in registry:
        context += f"Function: {func['name']}\n"
        context += f"Description: {func['description']}\n"
        context += "Parameters:\n"

        for param_name, param_info in func["parameters"].items():
            context += f"- {param_name}: {param_info['type']}\n"

        context += "\n"

    context += f"User request: {prompt}\n"
    return context
