from __future__ import annotations

import logging
from pathlib import Path

from griptape_statemachine.machines.base_machine import BaseMachine

logger = logging.getLogger(__name__)


class JokeMachine(BaseMachine):
    def start_machine(self) -> None:
        self.send("next_state")

    def on_enter_tell_joke(self) -> None:
        self.joke = self.get_structure("joke_teller").run("Tell a joke").output.value
        self.send("next_state")

    def on_enter_explain_joke(self) -> None:
        self.get_structure("joke_explainer").run("Explain this joke: " + self.joke)
        self.send("next_state")


config_file = Path(__file__).resolve().parent / "config.yaml"
machine = JokeMachine.from_config_file(str(config_file))
machine.start_machine()
