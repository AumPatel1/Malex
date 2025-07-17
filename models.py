"""
This class provides methods for building the model, making predictions, and inspecting its state, metadata, and metrics.

Key Features:
- Intent: A natural language description of the model's purpose.
- Input/Output Schema: Defines the structure and types of inputs and outputs.
- Mutable State: Tracks the model's lifecycle, training metrics, and metadata.
- Build Process: Integrates solution generation with callbacks.
- Chain of Thought: Captures the reasoning steps of the model building process.

Example:
>>>    model = Model(
>>>        intent="Given a dataset of house features, predict the house price.",
>>>        output_schema=create_model("output", **{"price": float}),
>>>        input_schema=create_model("input", **{
>>>            "bedrooms": int,
>>>            "bathrooms": int,
>>>            "square_footage": float
>>>        })
>>>    )
>>>
>>>    model.build(
>>>        datasets=[pd.read_csv("houses.csv")], 
>>>        provider="openai:gpt-4o-mini", 
>>>        max_iterations=10,
>>>        chain_of_thought=True  # Enable chain of thought logging
>>>    )
>>>
>>>    prediction = model.predict({"bedrooms": 3, "bathrooms": 2, "square_footage": 1500.0})
>>>    print(prediction)
"""
