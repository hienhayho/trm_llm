# TRM-LLM Architecture

```mermaid
flowchart TB
    subgraph Input
        tokens[Input Tokens]
        embed[Embedding Layer]
        tokens --> embed
    end

    subgraph Encoder
        transformer[Transformer Encoder]
        embed --> transformer
        transformer --> x[x: Encoded Input]
    end

    subgraph DeepSupervision ["Deep Supervision Loop ×T"]
        direction TB

        reasoning[Recursive Reasoning Module]
        action[Action State Module]
        heads[Output Heads]

        x --> reasoning
        y_in[y] --> reasoning
        z_in[z] --> reasoning

        reasoning --> z_out[z refined]
        z_out --> action
        y_in --> action
        action --> y_out[y updated]

        y_out --> heads

        heads --> action_out[Action Logits]
        heads --> q[Q Logit]

        y_out -.->|detach| y_in
        z_out -.->|detach| z_in
    end

    subgraph Generation
        gen_head[Generation Head]
        y_out --> gen_head
        x --> gen_head
        action_out --> gen_head
    end

    subgraph Output
        gen_head --> tool_call[Tool Call JSON]
        gen_head --> response[Direct Answer]
    end

    classDef loop fill:#e8f5e9,stroke:#1b5e20
    class reasoning,action,heads loop
```

## Data Flow

```
Input → Encoder → [Reasoning(x,y,z) → Action(y,z)]×T → Output Heads → Generation
```

## Components

| Module | Function |
|--------|----------|
| **Encoder** | Transforms input tokens to hidden states x |
| **Reasoning Module** | Recursively refines z = f(x, y, z) for n iterations |
| **Action Module** | Updates action state y = g(y, z) |
| **Output Heads** | Predicts action type, num calls, Q (correctness prediction) |
| **Generation Head** | Generates tool call JSON or direct answer text |
