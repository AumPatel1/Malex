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
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Dict,List,Type,Any
from deprecated import deprecated 

import pandas as pd
from pydantic import BaseModel

from Malex.callbacks import Callback
from Malex.core.interfaces.predictor import Predictor
from Malex.core.object_registry import ObjectRegistry
from Malex.core.state import ModelState  # Import from core package
from Malex.datasets import DatasetGenerator
from Malex.internal.common.datasets.interface import Dataset
from Malex.internal.common.provider import ProviderConfig
from Malex.internal.common.utils.model_utils import calculate_model_size, format_code_snippet
from Malex.internal.common.utils.pydantic_utils import map_to_basemodel, format_schema
from Malex.internal.models.entities.artifact import Artifact
from Malex.internal.models.entities.description import (
    ModelDescription,
    SchemaInfo,
    ImplementationInfo,
    PerformanceInfo,
    CodeInfo,
)
from Malex.internal.models.entities.metric import Metric

logger = logging.getlogger("__name__")

class Model:
    def __init__(
            self,
            intent:str,
            input:Type(BaseModel) | Dict[str,type] = None,
            output:Type(BaseModel) | Dict[str,type] = None,
            distributed:bool=False,
    ):
        """
     
        Initialise a model with a natural language description of its intent, as well as
        structured definitions of its input schema and output schema.

        :param intent: A human-readable, natural language description of the model's expected intent.
        :param input_schema: a pydantic model or dictionary defining the input schema
        :param output_schema: a pydantic model or dictionary defining the output schema
        :param distributed: Whether to use distributed training with Ray if available.
        
        """
        # The model's identity is defined by these fields
        self.intent: str = intent
        self.input_schema: Type[BaseModel] = map_to_basemodel("in", input_schema) if input_schema else None
        self.output_schema: Type[BaseModel] = map_to_basemodel("out", output_schema) if output_schema else None
        self.training_data: Dict[str, Dataset] = dict()
        self.distributed: bool = distributed

          # The model's mutable state is defined by these fields
        self.state: ModelState = ModelState.DRAFT
        self.predictor: Predictor | None = None
        self.trainer_source: str | None = None
        self.predictor_source: str | None = None
        self.feature_transformer_source: str | None = None
        self.dataset_splitter_source: str | None = None
        self.testing_source: str | None = None
        self.evaluation_report: Dict | None = None
        self.artifacts: List[Artifact] = []
        self.metric: Metric | None = None
        self.metadata: Dict[str, Any] = dict()  # todo: initialise metadata, etc

        # Registries used to make datasets, artifacts and other objects available across the system
        self.object_registry = ObjectRegistry()

        # Registries used to make datasets, artifacts and other objects available across the system
        self.object_registry = ObjectRegistry()

        # Setup the working directory and unique identifiers
        self.identifier: str = f"model-{abs(hash(self.intent))}-{str(uuid.uuid4())}"
        self.run_id = f"run-{datetime.now().isoformat()}".replace(":", "-").replace(".", "-")
        self.working_dir = f"./workdir/{self.run_id}/"
        os.makedirs(self.working_dir, exist_ok=True)

    @deprecated(reason = "Use model.build() instead",veresion="23.0.1")
    def build(
        self,
        Datasets : List[pd.DataFrame | DatasetGenerator],
        provider : str | ProviderConfig = "openai/gpt-4o-mini",
        imeout: int = None,
        max_iterations: int = None,
        run_timeout: int = 1800,
        callbacks: List[Callback] = None,
        verbose: bool = False,
        # resume: bool = False,
        enable_checkpointing: bool = False,
    ) -> None:
        """
        Build the model using the provided dataset and optional data generation configuration.

        DEPRECATED: This interface is deprecated. Use ModelBuilder.build() instead:

            from plexe import ModelBuilder
            builder = ModelBuilder(provider=provider, verbose=verbose, distributed=distributed)
            model = builder.build(intent=intent, datasets=datasets, ...)

        :param datasets: the datasets to use for training the model
        :param provider: the provider to use for model building, either a string or a ProviderConfig
                         for granular control of which models to use for different agent roles
        :param timeout: maximum total time in seconds to spend building the model (all iterations combined)
        :param max_iterations: maximum number of iterations to spend building the model
        :param run_timeout: maximum time in seconds for each individual model training run
        :param callbacks: list of callbacks to notify during the model building process
        :param verbose: whether to display detailed agent logs during model building (default: False)
        :param enable_checkpointing: whether to enable automatic checkpointing (default: True)
        :return:
        """
        warnings.warn(
            "Model.build() is deprecated. Use ModelBuilder.build() instead:\n\n"
            "    from plexe import ModelBuilder\n"
            "    builder = ModelBuilder(provider=provider, verbose=verbose, distributed=distributed)\n"
            "    model = builder.build(intent=intent, datasets=datasets, ...)\n",
            DeprecationWarning,
            stacklevel=2,
        )   
            # Import here to avoid circular dependency
        from plexe.model_builder import ModelBuilder


        )