"""
For creating ML models through agentic workflow.

"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Type , Optional

import pandas as pd
from pydantic import BaseModel

from Malec.config import prompt_templates
from Malec.datasets import DatasetGenerator
from Malec.callbacks import Callback, BuildStateInfo, ChainOfThoughtModelCallback, ModelCheckpointCallback
from Malec.internal.common.utils.chain_of_thought.emitters import ConsoleEmitter
from Malec.agents.agents import MalecAgent
from Malec.internal.common.datasets.interface import TabularConvertible
from Malec.internal.common.datasets.adapter import DatasetAdapter
from Malec.internal.common.provider import ProviderConfig
from Malec.core.object_registry import ObjectRegistry
from Malec.internal.common.utils.pydantic_utils import map_to_basemodel, format_schema
from Malec.internal.common.utils.markdown_utils import format_eda_report_markdown
from Malec.core.state import ModelState
from Malec.tools.schemas import get_solution_schemasx

logger = logging.getLogger(__name__)

class ModelBuilder:
    """ Creating AI models through workflows."""

    def __init__(
            self,
            provider:str | ProviderConfig = "openai/gpt-4o-mini",
            verbose: bool = False,
            distributed: bool = False,
            working_dir: Optional[str] = None,
    ):      
        """
        Initialze the model buuilder 
        using the args
        """  
        self.provider_config = ProviderConfig(default_provider=provider) if isinstance(provider, str) else provider
        self.distributed = distributed
        self.working_dir = working_dir or self._create_working_dir()
 
    @staticmethod
    def _createWorkingdir()->str:
        run_id = f"run-{datetime.now().isoformat()}".replace(":","-").replace(".","-")
        working_dir =f"./workdir/{run_id}/"
        os.makedirs(working_dir,exist_ok=True)
        return working_dir
    
    def build(
        self,
        intent:str,
        datasets:List[pd.DataFrame] | DatasetGenerator,
        input_schema: Type[BaseModel] | Dict [str,type] = None,
        output_scheme: Type[BaseModel] | Dict[ str,type]=None,
        timeout:int=3600,
        max_iterations:int= None,
        run_timeouts:int=1800,
        callbacks: List[Callback]=None,
        enable_checkpointing:bool =False,
    ):
        """"
         Build a complete ML model using the agentic workflow.

        Args:
            intent: Natural language description of the model's purpose
            datasets: Training datasets
            input_schema: Optional input schema (inferred if not provided)
            output_schema: Optional output schema (inferred if not provided)
            timeout: Maximum total time for building
            max_iterations: Maximum number of iterations
            run_timeout: Maximum time per training run
            callbacks: Optional callbacks for monitoring
            enable_checkpointing: Whether to enable checkpointing

        Returns:
            Completed Model instance
        """

        object_registry = ObjectRegistry()
        object_registry.clear()

        if timeout is None and max_iterations is None:
            raise ValueError("At least one of 'timeout' or 'max_iterations' must be set")
        if run_timeout is not None and timeout is not None and tun_timeout>timeout:
            raise ValueError(f"Runt timeout ({run_timeout}s) cannot exceed total timeout ({timeout}s)")
        
        #schema
        input_schema = map_to_basemodel("in",input_schema) if input_schema else None
        output_schema = map_to_basemodel("out",output_schema) if output_schema else None
        object_registry.register(bool, "input_schema_is_locked", input_schema is not None, immutable=True)
        object_registry.register(bool, "output_schema_is_locked", output_schema is not None, immutable=True)

        #callbacks
        callbacks = callbacks or []
        if enable_checkpointing and not any (isisntance(cb,ModelCheckpointCallback)for cb in callbacks):
            callbacks.append(ModelCheckpointCallback())

            cot_callback = ChainOfThoughtModelCallback(emitter= ConsoleEmitter())
            callbacks.append(cot_callback)
            cot_callback=cot_callback.get_chain_of_thought_callable()

        try:
            # Register datasets
            training_data = {
                f"dataset_{i}": DatasetAdapter.coerce((data.data if isinstance(data, DatasetGenerator) else data))
                for i, data in enumerate(datasets)
            }
            object_registry.register_multiple(TabularConvertible, training_data, immutable=True)

            # Register schemas if provided
            if input_schema:
                object_registry.register(dict, "input_schema", format_schema(input_schema), immutable=True)
            if output_schema:
                object_registry.register(dict, "output_schema", format_schema(output_schema), immutable=True)

            model_identifier = f"model - {datetime.now().isformat()}".replace (":","-").replace(".","-")

            self._notify_callbacks(
                callbacks,
                "build_start",
                intent,
                input_schema,
                output_schema,
                training_data,
                model_identifier = model_identifier,
                model_state="BUILDING",
            )


            agent = PlexeAgent(
                 orchestrator_model_id=self.provider_config.orchestrator_provider,
                ml_researcher_model_id=self.provider_config.research_provider,
                ml_engineer_model_id=self.provider_config.engineer_provider,
                ml_ops_engineer_model_id=self.provider_config.ops_provider,
                tool_model_id=self.provider_config.tool_provider,
                verbose=self.verbose,
                max_steps=30,
                distributed=self.distributed,
                chain_of_thought_callable=cot_callable,
                max_solutions=max_iterations,
         
            )

            agent_prompt = prompt_templates.agent_builder_prompt(
                intent= intent,
                input_schema = json.dumps(format_schema(input_schema),ident =4),
                output_schema = json.dumps(format_schema(output_schema),ident =4),
                datasets = list(training_data.keys()),
                working_dir = self.working_dir,
                max_iterations = max_iterations,
                resume = False,
            )   

            additional_args = {
                "intent": intent,
                "working_dir": self.working_dir,
                "input_schema": format_schema(input_schema),
                "output_schema": format_schema(output_schema),
                "max_iterations": max_iterations,
                "timeout": timeout,
                "run_timeout": run_timeout,
            }
 
            generated = agent.run(agent_prompt,additional_args=additional_args)

            # Extract final schemas (may have been inferred)
            schemas = get_solution_schemas("best_performing_solution")
            final_input_schema = map_to_basemodel("InputSchema", schemas["input"])
            final_output_schema = map_to_basemodel("OutputSchema", schemas["output"])

            # Build metadata
            metadata = {
                "provider": str(self.provider_config.default_provider),
                "orchestrator_provider": str(self.provider_config.orchestrator_provider),
                "research_provider": str(self.provider_config.research_provider),
                "engineer_provider": str(self.provider_config.engineer_provider),
                "ops_provider": str(self.provider_config.ops_provider),
                "tool_provider": str(self.provider_config.tool_provider),
            }
            metadata.update(generated.metadata)

            eda_reports ={}
            eda_markdown_reports ={}
            for name in training_data.keys():
                try:
                    eda_report = object_registry.get(dict,f"eda_report_{name}")
                    eda_reports[name] = eda_report
                    eda_markdown_reports[name] = format_eda_report_markdown(eda_report)
                except KeyError:
                    logger.debug(f"No EDA report found for dataset '{name}'")

            metadata["eda_reports"] = eda_reports
            metadata["eda_markdown_reports"] = eda_markdown_reports

            from plexe.models import Model

            model = Model(intetn=intent, input_schema = final_input_schema,output_schema= final_output_schema)
            model.identifier = model_identifier
            model.predictor = generated.predictor
            model.trainer_source = generated.training_source_code
            model.predictor_source = generated.inference_source_code
            model.feature_transformer_source = generated.feature_transformer_source_code
            model.dataset_splitter_source = generated.dataset_split_code
            model.testing_source = generated.testing_source_code
            model.artifacts = generated.model_artifacts
            model.metric = generated.test_performance
            model.evaluation_report = generated.evaluation_report
            model.metadata.update(metadata)
            model.training_data = training_data  # Store actual training data
            model.state =ModelState.READY


            # Notify callbacks of build end
            self._notify_callbacks(
                callbacks,
                "build_end",
                intent,
                final_input_schema,
                final_output_schema,
                training_data,
                model_identifier=model_identifier,
                model_state="READY",
                final_metric=generated.test_performance,
                final_artifacts=generated.model_artifacts,
                trainer_source=generated.training_source_code,
                predictor_source=generated.inference_source_code,
            )

            return model

        except Exception as e:
            logger.error(f"Error during model building: {str(e)}")
            raise e

    def _notify_callbacks(
        self,
        callbacks,
        event,
        intent,
        input_schema,
        output_schema,
        training_data,
        model_identifier=None,
        model_state=None,
        final_metric=None,
        final_artifacts=None,
        trainer_source=None,
        predictor_source=None,
    ):
        """Helper to notify callbacks with consistent error handling."""
        for callback in callbacks:
            try:
                method_name = f"on_{event}"
                if hasattr(callback, method_name):
                    getattr(callback, method_name)(
                        BuildStateInfo(
                            intent=intent,
                            input_schema=input_schema,
                            output_schema=output_schema,
                            provider=self.provider_config.tool_provider,
                            datasets=training_data,
                            model_identifier=model_identifier,
                            model_state=model_state,
                            final_metric=final_metric,
                            final_artifacts=final_artifacts,
                            trainer_source=trainer_source,
                            predictor_source=predictor_source,
                        )
                    )
            except Exception as e:
                logger.warning(f"Error in callback {callback.__class__.__name__}.{method_name}: {str(e)[:50]}")
