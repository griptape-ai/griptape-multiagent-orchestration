from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from goap_base_machine import GoapBaseMachine
from griptape.memory.structure import ConversationMemory
from griptape.utils import StructureVisualizer

if TYPE_CHECKING:
    from griptape.tools import BaseTool


class GoapMachine(GoapBaseMachine):
    """State machine with GOAP"""

    @property
    def tools(self) -> dict[str, BaseTool]:
        return {}

    def get_prompt_entry_by_id(self, prompt_id: str) -> dict[str, str]:
        prompts = self.config["prompts"]
        return prompts[prompt_id]

    def start_machine(self) -> None:
        """Starts the machine."""
        # Clear input history.
        self.latest_user_input = None
        self.send("go_to_first_state")

    def on_enter_generic(self) -> None:
        # Kick off by evaluating what to say here.
        # Clear conversation memory locally
        # for structure in self._structures:
        #     if "state_goal" in structure:
        #         self._structures[structure].conversation_memory = ConversationMemory()
        # Clear conversation memory in the cloud (delete the thread)
        for assistant in self._assistants:
            if "state_goal" in assistant:
                self.reset_conversation_memory(assistant)
        self.eval_current_situation_workflow(None, None)

    def on_enter_end(self) -> None:
        self.outputs_to_user.append("Operation Complete.")

    def on_event_anystate(self, event_: dict) -> None:  # noqa: PLR0912, PLR0915
        event_source = event_["type"]
        event_value = event_["value"]

        # Did the event come from the user, or from a Griptape Structure?
        match event_source:
            case "griptape_event":
                event_type = event_value["type"]

                # Now see exactly which type of Griptape event it was.
                match event_type:
                    case "FinishStructureRunEvent":
                        # OK now who is it from?
                        structure_id = event_value["structure_id"]
                        match structure_id:
                            case "presenter":
                                # Save off the response so that we can feed it to the evaluator to assess.
                                self.most_recent_presenter_response = event_value[
                                    "output_task_output"
                                ]["value"]
                                # Echo the response directly back to the user.
                                self.outputs_to_user.append(
                                    self.most_recent_presenter_response
                                )
                            case "goal_workflow":
                                # The workflow has completed, which means the goals have been ran
                                # Will return a json that has the Goals as several types
                                goals = json.loads(
                                    event_value["output_task_output"]["value"]
                                )

                                possible_states = self.state_transitions[
                                    self.current_state_value
                                ]
                                possible_states["stay"] = (
                                    "general_state_goal goals_fulfilled is 'no'"
                                )
                                prompt_entry = self.get_prompt_entry_by_id(
                                    "state_changer_instructions"
                                )
                                prompt = prompt_entry["prompt"]
                                # This is dependent on the format of the prompt.
                                prompt = prompt.format(
                                    state_transition_dict=possible_states, goals=goals
                                )
                                # self.get_structure("state_changer").run(prompt)
                                self.get_assistant("state_changer", prompt)
                            # Check to see which state to stay in or go to.
                            case "state_changer":
                                goals = json.loads(
                                    event_value["output_task_output"]["value"]
                                )
                                if "stay" in goals:
                                    missing_requirements = goals["stay"]
                                    prompt = self.get_prompter_prompt(
                                        missing_requirements
                                    )
                                    # self.get_structure("prompter").run(prompt)
                                    self.get_assistant("prompter", prompt)
                                else:
                                    state_name = list(goals.keys())[0]
                                    self.send(state_name)
                            case "prompter":
                                steer = json.loads(
                                    event_value["output_task_output"]["value"]
                                )
                                steering_prompt = steer["steering_prompt"]
                                templated_prompt_entry = self.get_prompt_entry_by_id(
                                    "instructions_to_presenter_to_steer_user"
                                )
                                templated_prompt = templated_prompt_entry["prompt"]
                                output_prompt = templated_prompt.format(
                                    steering_prompt=steering_prompt
                                )

                                # Append the history.
                                if self.latest_user_input:
                                    templated_prompt_entry = self.get_prompt_entry_by_id(
                                        "presenter_add_user_conversation_to_conversation_memory"
                                    )
                                    templated_prompt = templated_prompt_entry["prompt"]
                                    append = templated_prompt.format(
                                        latest_user_input=self.latest_user_input
                                    )
                                    output_prompt += f"\n{append}"

                                # Ask the conversational agent to generate a prompt that steers us.
                                # self.get_structure("presenter").run(output_prompt)
                                self.get_assistant("presenter", output_prompt)
                            case _:
                                pass
                                # self.outputs_to_user.append(
                                #     f"***DEBUG: Got an event from structure: {structure_id}"
                                # )
                    case _:
                        pass
            case "user_input":
                # See if, based on this input, what remains
                self.latest_user_input = event_value
                self.eval_current_situation_workflow(
                    latest_agent_response=self.most_recent_presenter_response,
                    latest_user_input=self.latest_user_input,
                )
            case _:
                err_msg = f"""Unexpected Transition Event ID: {event_value}."""
                raise ValueError(err_msg)

    def eval_current_situation_workflow(
        self, latest_agent_response: str | None, latest_user_input: str | None
    ) -> None:
        prompt_entry = self.get_prompt_entry_by_id("instructions_for_goalie")
        prompt = prompt_entry["prompt"]
        # This can all stay the same, because these rules should be the same for these goal oriented agents.
        if latest_agent_response:
            templated_prompt_entry = self.get_prompt_entry_by_id(
                "goalie_add_presenter_conversation_to_conversation_memory"
            )
            templated_prompt = templated_prompt_entry["prompt"]
            append = templated_prompt.format(
                latest_agent_response=latest_agent_response
            )
            prompt += f"\n{append}"

        if latest_user_input:
            templated_prompt_entry = self.get_prompt_entry_by_id(
                "goalie_add_user_conversation_to_conversation_memory"
            )
            templated_prompt = templated_prompt_entry["prompt"]
            append = templated_prompt.format(latest_user_input=latest_user_input)
            prompt += f"\n{append}"
        # Get the workflow. This creates it with all of the structures that have "goal" in the name.
        # self.get_goal_workflow(prompt).run()
        self.get_goal_workflow_assistants(prompt).run()

    def get_prompter_prompt(self, missing_requirements: str) -> str:
        templated_prompt_entry = self.get_prompt_entry_by_id(
            "prompter_goalie_unfulfilled_make_judgment_call"
        )
        templated_prompt = templated_prompt_entry["prompt"]
        output_prompt = templated_prompt.format(
            missing_requirements=missing_requirements
        )
        # If we have recent input, append that.
        if self.latest_user_input:
            templated_prompt_entry = self.get_prompt_entry_by_id(
                "prompter_add_user_conversation_to_conversation_memory"
            )
            templated_prompt = templated_prompt_entry["prompt"]
            append = templated_prompt.format(latest_user_input=self.latest_user_input)
            output_prompt += f"\n{append}"

            # Generate a prompt that steers us.
        return output_prompt


if __name__ == "__main__":
    pass
    # config_path = Path.cwd().joinpath(Path("examples/goap_machine/config.yaml"))
    # machine = GoapMachine.from_config_file(config_path)
    # machine.start_machine()
    # structure = machine.get_goal_workflow_assistants("hi")
    # print(StructureVisualizer(structure).to_url())
