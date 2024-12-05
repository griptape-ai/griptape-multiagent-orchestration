from __future__ import annotations

import json
import logging
from multiprocessing import Value
import time
from httpx import HTTPStatusError
import requests
import os
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from dotenv import load_dotenv


from griptape.drivers import (
    OpenAiChatPromptDriver,
    LocalStructureRunDriver,
    GriptapeCloudEventListenerDriver,
)
from griptape.events import BaseEvent, EventBus, EventListener, FinishStructureRunEvent
from griptape.rules import Rule, Ruleset
from griptape.structures import Agent, Workflow
from griptape.tasks import (
    PromptTask,
    StructureRunTask,
    CodeExecutionTask,
)
from griptape.utils import dict_merge
from griptape.memory.structure import ConversationMemory
from griptape.artifacts import TextArtifact
from statemachine import State, StateMachine
from statemachine.factory import StateMachineMetaclass

from griptape_statemachine.parsers import GoapConfigParser

logger = logging.getLogger(__name__)
logging.getLogger("griptape").setLevel(logging.ERROR)

if TYPE_CHECKING:
    from griptape.structures import Structure
    from griptape.tools import BaseTool
    from statemachine.event import Event

load_dotenv()


class GoapBaseMachine(StateMachine):
    """Base class for a machine.


    Attributes:
        config_file (Path): The path to the configuration file.
        config (dict): The configuration data.
        outputs_to_user (list[str]): Outputs to return to the user.
    """

    def __init__(self, config_file: Path, **kwargs) -> None:
        self.config_parser = GoapConfigParser(config_file)
        self.config = self.config_parser.parse()
        self.outputs_to_user: list[str] = []
        self.latest_user_input: str | None = None
        self.state_transitions: dict[str, str] = self.__class__.state_transitions
        self.most_recent_presenter_response: str | None = None

        self.state_status: dict[str, bool] = {}

        for key in self.state_transitions.keys():
            self.state_status[key] = False

        # self._structures: dict[str, Agent] = {}

        # All new for moving to cloud
        self._rulesets: dict[str, dict[str, str]] = {"state": {}, "global": {}}
        self._assistants: dict[str, str] = {}
        self._threads: dict[str, str] = {}

        headers = {
            "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        url = "https://cloud.griptape.ai/api/assistants"
        response = requests.get(url=url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            for assistant in data["assistants"]:
                self._assistants[assistant["name"]] = assistant["assistant_id"]

        def on_event(event: BaseEvent) -> None:
            print(f"Received Griptape event: {json.dumps(event.to_dict(), indent=2)}")
            try:
                self.send(
                    "process_event",
                    event_={"type": "griptape_event", "value": event.to_dict()},
                )
            except Exception as e:
                errormsg = f"Would not allow process_event to be sent. Check to see if it is defined in the config.yaml. Error:{e}"
                raise ValueError(errormsg) from e

        # TODO: Do i need to change this to push to gtc?
        EventBus.clear_event_listeners()
        EventBus.add_event_listener(
            EventListener(on_event, event_types=[FinishStructureRunEvent]),
        )

        super().__init__()

    @property
    def available_events(self) -> list[str]:
        return self.current_state.transitions.unique_events

    @property
    @abstractmethod
    def tools(self) -> dict[str, BaseTool]:
        """Returns the Tools for the machine."""
        ...

    @property
    def _current_state_config(self) -> dict:
        return self.config["states"][self.current_state_value]

    @classmethod
    def from_definition(  # noqa: C901, PLR0912
        cls, definition: dict, **extra_kwargs
    ) -> GoapBaseMachine:
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
        try:
            states_instances = {}
            for state_id, state_kwargs in definition["states"].items():
                if state_id not in (
                    "start",
                    "end",
                ):
                    states_instances[state_id] = State(
                        **state_kwargs, value=state_id, enter="on_enter_generic"
                    )
                else:
                    states_instances[state_id] = State(**state_kwargs, value=state_id)
        except Exception as e:
            errormsg = f"""Error in state definition: {e}.
            """
            raise ValueError(errormsg) from e

        events = {}
        state_transitions = {}
        for event_name, transitions in definition["events"].items():
            for transition_data in transitions:
                try:
                    source_name = transition_data["from"]
                    source = states_instances[source_name]
                    target = states_instances[transition_data["to"]]
                    relevance = ""
                    if "relevance" in transition_data:
                        relevance = transition_data["relevance"]
                    if source_name not in state_transitions:
                        state_transitions[source_name] = {event_name: relevance}
                    else:
                        state_transitions[source_name][event_name] = relevance
                except Exception as e:
                    errormsg = f"Error:{e}. Please check your transitions to be sure each transition has a source and destination."
                    raise ValueError(errormsg) from e

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
        for state_id, state in states_instances.items():
            if state_id != "end":
                transition = state.to(
                    state,
                    event="process_event",
                    on="on_event_anystate",
                    internal=True,
                )
                if "process_event" in events:
                    events["process_event"] |= transition
                else:
                    events["process_event"] = transition

        attrs_mapper = {
            **extra_kwargs,
            **states_instances,
            **events,
            "state_transitions": state_transitions,
        }

        return cast(
            GoapBaseMachine,
            StateMachineMetaclass(cls.__name__, (cls,), attrs_mapper)(**extra_kwargs),
        )

    @classmethod
    def from_config_file(
        cls,
        config_file: Path,
        **extra_kwargs,
    ) -> GoapBaseMachine:
        """Creates a StateMachine class from a configuration file"""
        config_parser = GoapConfigParser(config_file)
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
        print(f"Transitioning from {source} to {state} with event {event}")

    def get_structure(self, structure_id: str) -> Structure:
        global_structure_config = self.config["assistants"][structure_id]
        state_structure_config = self._current_state_config.get("assistants", {}).get(
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
        structure = Agent(
            id=structure.id,
            prompt_driver=structure.prompt_driver,
            conversation_memory=structure.conversation_memory,
            rulesets=[
                *self._get_structure_rulesets(structure_config.get("ruleset_ids", [])),
            ],
        )
        print(f"Structure: {structure_id}")
        for ruleset in structure.rulesets:
            for rule in ruleset.rules:
                print(f"Rule: {rule.value}")
        return structure

    def get_goal_workflow(self, input_value: str) -> Structure:
        structure = Workflow(
            id="goal_workflow",
        )
        end_task = PromptTask(
            f"""Take in the {{{{parent_outputs}}}} and combine them into a json file and return it:
                """,
            id="end_task",
            rules=[
                Rule("No markdown, no backticks, no code."),
                Rule("Give the output exactly as it came with no extra commentary."),
                Rule("Only return one json file."),
                Rule("If there is no parent_outputs, return an empty json file."),
            ],
        )
        try:
            for goal_structure in self.config["states"][self.current_state_value][
                "assistants"
            ]:
                # Must have state_goal in it to be chosen.
                # TODO: Is there a way to do this with assistants?
                if "state_goal" in goal_structure:
                    run_task = StructureRunTask(
                        input=(f"{input_value}",),
                        id=goal_structure,
                        # This will be something that has to change moving to cloud.
                        driver=LocalStructureRunDriver(
                            structure_factory_fn=lambda gs=goal_structure: self.get_structure(
                                gs
                            )
                        ),
                        child_ids=["end_task"],
                    )
                    structure.add_task(run_task)
                    end_task.add_parent(run_task)
        except KeyError:
            pass
        # Add the end task that consolidates everything
        structure.add_task(end_task)
        return structure

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

    # EVERYTHING BELOW THIS LINE IS SPECIFICALLY FOR ASSISTANTS
    def _get_structure_rulesets_ids(
        self, ruleset_ids: list[str], global_rule: str
    ) -> list[str]:
        headers = {
            "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        ruleset_gtc_ids = []
        for ruleset_id in ruleset_ids:
            if ruleset_id not in self._rulesets[global_rule]:
                # Create the rules and the ruleset and store it
                # Get URL
                rules = self.config["rulesets"][ruleset_id]
                rules_gtc_ids = []
                # Make each rule
                for rule in rules["rules"]:
                    url = "https://cloud.griptape.ai/api/rules"
                    rule_response = requests.post(
                        url=url,
                        headers=headers,
                        json={
                            "name": f"{rules['name']} Rule {len(rules_gtc_ids)}",
                            "rule": rule,
                        },
                    )
                    if rule_response.status_code == 201:
                        data = rule_response.json()
                        rules_gtc_ids.append(data["rule_id"])
                    else:
                        errormsg = f"Unable to create Rule: {rule_response}"
                        raise ValueError(errormsg)
                url = "https://cloud.griptape.ai/api/rulesets"
                ruleset_response = requests.post(
                    url=url,
                    headers=headers,
                    json={"name": rules["name"], "rule_ids": rules_gtc_ids},
                )
                if ruleset_response.status_code == 201:
                    data = ruleset_response.json()
                    self._rulesets[global_rule][ruleset_id] = data["ruleset_id"]
                else:
                    errormsg = f"Unable to create Ruleset: {ruleset_response}"
                    raise ValueError(errormsg)
            ruleset_gtc_ids.append(self._rulesets[global_rule][ruleset_id])
        return ruleset_gtc_ids

    def get_assistant(self, assistant_english_id: str, input: str) -> str:
        # Create an assistant with the rules that we want and get the structure ID.
        headers = {
            "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        if assistant_english_id not in self._assistants:
            # Initialize Assistant with all the expensive setup and the global rulesets
            global_structure_config = self.config["assistants"][assistant_english_id]
            url = "https://cloud.griptape.ai/api/assistants"
            response = requests.post(
                url=url,
                headers=headers,
                json={
                    "name": assistant_english_id,
                    "ruleset_ids": [
                        *self._get_structure_rulesets_ids(
                            global_structure_config.get("ruleset_ids", []), "global"
                        )
                    ],
                },
            )
            # If the assistant is successfully created, add it to the assistants
            if response.status_code == 201:
                data = response.json()
                assistant_id = data["assistant_id"]
                self._assistants[assistant_english_id] = assistant_id
            else:
                # Otherwise, raise an error
                errormsg = f"Error creating Assistant: {response}"
                raise ValueError(errormsg)
        # Create the Assistant Run with input and state specific rulesets.
        state_structure_config = self._current_state_config.get("assistants", {}).get(
            assistant_english_id, {}
        )
        url = f"https://cloud.griptape.ai/api/assistants/{self._assistants[assistant_english_id]}/runs"
        response = requests.post(
            url=url,
            headers=headers,
            json={
                "input": input,
                "thread_id": self.get_thread(assistant_english_id),
                "additional_ruleset_ids": [
                    *self._get_structure_rulesets_ids(
                        state_structure_config.get("ruleset_ids", []), "state"
                    )
                ],
            },
        )
        if response.status_code == 201:
            print("Made a successful assistant call")
            data = response.json()
            assistant_run_id = data["assistant_run_id"]
            output = self.poll_assistants(assistant_run_id)
            print("Finished polling assistants")
            # Manually edit - IS THIS BAD!!!
            output["structure_id"] = assistant_english_id
            self.send(
                "process_event",
                event_={"type": "griptape_event", "value": output},
            )
            print("returning ", output["output_task_output"]["value"])
            return output["output_task_output"]["value"]
        else:
            return "Failed to create assistant run"

    def poll_assistants(self, assistant_run_id: str) -> dict[Any, Any] | Literal[""]:
        response = self._list_events(0, assistant_run_id)
        events = response.json()["events"]
        offset = response.json()["next_offset"]
        not_finished = True
        output = ""
        while not_finished:
            time.sleep(0.5)
            for event in events:
                if event["type"] == "FinishStructureRunEvent":
                    not_finished = False
                    output = dict(event["payload"])
                    break
            response = self._list_events(offset, assistant_run_id)
            response.raise_for_status()
            events = response.json()["events"]
            offset = response.json()["next_offset"]

        return output

    # Create goal workflow with CET instead of StructureRunTasks
    def get_goal_workflow_assistants(self, input_value: str) -> Structure:
        structure = Workflow(
            id="goal_workflow",
        )
        end_task = PromptTask(
            f"""Take in the {{{{parent_outputs}}}} and combine them into a json file and return it:
                """,
            id="end_task",
            rules=[
                Rule("No markdown, no backticks, no code."),
                Rule("Give the output exactly as it came with no extra commentary."),
                Rule("Only return one json file."),
                Rule("If there is no parent_outputs, return an empty json file."),
            ],
        )
        try:
            for goal_structure in self.config["states"][self.current_state_value][
                "assistants"
            ]:
                # Must have state_goal in it to be chosen.
                if "state_goal" in goal_structure:
                    prompt = json.dumps({"input": input_value, "id": goal_structure})
                    run_task = CodeExecutionTask(
                        # on_run=self.run_assistant_cet,
                        run_fn=self.run_assistant_cet,
                        input=prompt,
                        id=goal_structure,
                        child_ids=["end_task"],
                    )
                    structure.add_task(run_task)
                    end_task.add_parent(run_task)
        except KeyError:
            pass
        # Add the end task that consolidates everything
        structure.add_task(end_task)
        return structure

    # Creates the code execution task that runs the assistant
    def run_assistant_cet(self, task: CodeExecutionTask) -> TextArtifact:
        data = task.input.value  # Get the input value for the code execution task.
        data = json.loads(data)
        input_value = data["input"]
        assistant_id = data["id"]
        print("about to call get assistant")
        output_value = self.get_assistant(
            assistant_id, input_value
        )  # get the assistant and kick off the run
        print("Finished get assistant, ", output_value)
        # Now poll and wait until we get the output from the assistant.
        # get the output of this response
        return TextArtifact(output_value)

    # Calls "get" in order to get events that have been emitted!
    def _list_events(self, offset: int, assistant_run_id: str):
        url = f"https://cloud.griptape.ai/api/assistant-runs/{assistant_run_id}/events"
        headers = {
            "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            url=url, headers=headers, params={"offset": offset, "limit": 100}
        )
        response.raise_for_status()
        return response

    def get_thread(self, structure_id: str) -> str:
        # Check if there is conversation memory for this already
        if structure_id == "state_changer":
            return ""
        if structure_id not in self._threads:
            # Create a new thread based on this line of conversation memory
            url = "https://cloud.griptape.ai/api/threads"
            headers = {
                "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
                "Content-Type": "application/json",
            }
            json_body = {"name": f"{structure_id} for {self.current_state_value}"}
            response = requests.post(url=url, headers=headers, json=json_body)
            if response.status_code == 201:
                # Thread created successfully
                data = response.json()
                self._threads[structure_id] = data["thread_id"]
                # Give new thread back to the assistant for the run.
                return data["thread_id"]
            else:
                errormsg = f"Unsuccessful Thread Creation: {response}"
                raise ValueError(errormsg)
        # Return the current thread
        return self._threads[structure_id]

    # Used to reset the goal assistants conversation memory when entering a new state.
    def reset_conversation_memory(self, structure_id: str) -> None:
        if structure_id not in self._threads:
            return
        # Removes from the dictionary
        thread_id = self._threads[structure_id]
        url = f"https://cloud.griptape.ai/api/threads/{thread_id}"
        headers = {
            "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        response = requests.patch(
            url=url,
            headers=headers,
            json={
                "messages": [
                    {
                        "input": "You are moving to a new state. All memory before this was a different state.",
                        "output": "Anything that happened before this message cannot be used to fulfill a goal. It is context.",
                    }
                ]
            },
        )

        # thread_id = self._threads.pop(structure_id)
        # url = f"https://cloud.griptape.ai/api/threads/{thread_id}"
        # headers = {
        #             "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
        #             "Content-Type" : "application/json"
        #         }
        # response = requests.delete(url=url,headers=headers)
        # if response.status_code != 204:
        #     errormsg = f"Deleting thread unsuccessful: {response}"
        #     raise ValueError(errormsg)

    # Destroys all Rules, Rulesets, Threads. Doesn't destroy the assistants.
    def destroy_all_threads_and_rules(self) -> None:
        headers = {
            "Authorization": f"Bearer {os.environ['GT_CLOUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        # Delete all existing threads
        # for thread_id in self._threads.values():
        #     thread_url = f"https://cloud.griptape.ai/api/threads/{thread_id}"
        #     response = requests.delete(url=thread_url, headers=headers)
        #     if response.status_code != 204:
        #         errormsg = f"Deleting thread unsuccessful: {response.json()}"
        #         raise ValueError(errormsg)
        # Delete all existing rules and rulesets
        for ruleset_id in self._rulesets["state"].values():
            # Define Ruleset url
            ruleset_url = f"https://cloud.griptape.ai/api/rulesets/{ruleset_id}"
            # Get the ruleset to get rule ids
            response = requests.get(url=ruleset_url, headers=headers)
            # If accessed correctly, delete all rules in ruleset
            if response.status_code == 200:
                rule_ids = response.json()["rule_ids"]
                for rule_id in rule_ids:
                    rule_url = f"https://cloud.griptape.ai/api/rules/{rule_id}"
                    response = requests.delete(url=rule_url, headers=headers)
                    if response.status_code != 204:
                        errormsg = f"Deleting rule unsuccessful: {response}"
                        raise ValueError(errormsg)
            # Delete the ruleset
            response = requests.delete(url=ruleset_url, headers=headers)
            if response.status_code != 204:
                errormsg = f"Deleting ruleset unsuccessful: {response}"
                raise ValueError(errormsg)


# TODO:
# Use the new driver to perform a structure run task that subscribes to the eventbus.
def on_event(event):
    print(event)
    print("Ok it listened")


# if __name__ == "__main__":
#     EventBus.add_event_listener(
#         EventListener(on_event),
#     )
#     task = AssistantTask(
#         input=("Echo this back to me please"),
#         driver=GriptapeCloudAssistantDriver(
#             api_key=os.environ["GT_CLOUD_API_KEY"],
#             assistant_id="b8f88da7-51a4-4cd8-9093-09f224d5e1ee",
#             stream=False,
#         ),
#     )
#     task.run()
