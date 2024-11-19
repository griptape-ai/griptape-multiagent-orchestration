from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, cast

from griptape.drivers import OpenAiChatPromptDriver
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent
from griptape.utils import dict_merge
from statemachine import State, StateMachine
from statemachine.factory import StateMachineMetaclass
from pathlib import Path

from griptape_statemachine.parsers import ConfigParser

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from griptape.structures import Structure
    from statemachine.event import Event


class BaseMachine(StateMachine):
    """Base class for a machine.


    Attributes:
        config_file (str): The path to the configuration file.
    """

    def __init__(self, config_file: str) -> None:
        self.config_parser = ConfigParser(Path(config_file))
        self.config = self.config_parser.parse()
        self.last_user_input: str | None = None

        self._structures: dict[str, Agent] = {}

        super().__init__()

    @property
    def available_events(self) -> list[str]:
        return self.current_state.transitions.unique_events

    @property
    def _current_state_config(self) -> dict:
        return self.config["states"][self.current_state_value]

    @classmethod
    def from_definition(cls, definition: dict, **extra_kwargs) -> BaseMachine:
        """
        Creates a StateMachine class from a dictionary definition, using the StateMachineMetaclass metaclass.
        It maps the definition to the StateMachineMetaclass parameters and then creates the class.

        Example usage with a traffic light machine:

        >>> machine = BaseMachine.from_definition(
        ...     "TrafficLightMachine",
        ...     {
        ...         "states": {
        ...             "green": {"initial": True},
        ...             "yellow": {},
        ...             "red": {},
        ...         },
        ...         "events": {
        ...             "transitions": [
        ...                 {"from": "green", "to": "yellow"},
        ...                 {"from": "yellow", "to": "red"},
        ...                 {"from": "red", "to": "green"},
        ...             ]
        ...         },
        ...     }
        ... )

        """

        states_instances = {
            state_id: State(**state_kwargs, value=state_id)
            for state_id, state_kwargs in definition["states"].items()
        }

        events = {}
        for event_name, transitions in definition["events"].items():
            for transition_data in transitions:
                source = states_instances[transition_data["from"]]
                target = states_instances[transition_data["to"]]

                transition = source.to(
                    target,
                    event=event_name,
                    cond=transition_data.get("cond"),
                    unless=transition_data.get("unless"),
                    on=transition_data.get("on"),
                    internal=transition_data.get("internal"),
                )

                if event_name in events:
                    events[event_name] |= transition
                else:
                    events[event_name] = transition

        attrs_mapper = {**extra_kwargs, **states_instances, **events}

        return cast(
            BaseMachine,
            StateMachineMetaclass(cls.__name__, (cls,), attrs_mapper)(**extra_kwargs),
        )

    @classmethod
    def from_config_file(cls, config_file: str, **extra_kwargs) -> BaseMachine:
        """Creates a StateMachine class from a configuration file."""
        config_parser = ConfigParser(Path(config_file))
        config = config_parser.parse()
        extra_kwargs["config_file"] = config_file

        definition_states = {
            state_id: {
                "initial": state_value.get("initial", False),
                "final": state_value.get("final", False),
            }
            for state_id, state_value in config["states"].items()
        }
        definition_events = {
            event_name: list(event_value["transitions"])
            for event_name, event_value in config["events"].items()
        }
        definition = {"states": definition_states, "events": definition_events}

        return cls.from_definition(definition, **extra_kwargs)

    @abstractmethod
    def start_machine(self) -> None:
        """Starts the machine."""
        ...

    def reset_structures(self) -> None:
        """Resets the structures."""
        self._structures = {}

    def on_enter_state(self, source: State, state: State, event: Event) -> None:
        logger.info("Transitioning from %s to %s with event %s", source, state, event)

    def get_structure(self, structure_id: str) -> Structure:
        global_structure_config = self.config["structures"][structure_id]
        state_structure_config = self._current_state_config.get("structures", {}).get(
            structure_id, {}
        )
        structure_config = dict_merge(global_structure_config, state_structure_config)

        if structure_id not in self._structures:
            # Initialize Structure with all the expensive setup
            structure = Agent(
                id=structure_id,
                prompt_driver=OpenAiChatPromptDriver(
                    model=structure_config.get("model", "gpt-4o"),
                ),
            )
            self._structures[structure_id] = structure

        # Create a new clone with state-specific stuff
        structure = self._structures[structure_id]

        return Agent(
            id=structure.id,
            prompt_driver=structure.prompt_driver,
            conversation_memory=structure.conversation_memory,
            rulesets=[
                *self._get_structure_rulesets(structure_config.get("ruleset_ids", [])),
            ],
        )

    def _get_structure_rulesets(self, ruleset_ids: list[str]) -> list[Ruleset]:
        ruleset_configs = [
            self.config["rulesets"][ruleset_id] for ruleset_id in ruleset_ids
        ]

        # Convert ruleset configs to Rulesets
        return [
            Ruleset(
                name=ruleset_config["name"],
                rules=[Rule(rule) for rule in ruleset_config["rules"]],
            )
            for ruleset_config in ruleset_configs
        ]
