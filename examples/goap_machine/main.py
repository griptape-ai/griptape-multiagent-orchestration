from __future__ import annotations

import contextvars
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from statemachine.contrib.diagram import DotGraphMachine

from goap_machine import GoapMachine

load_dotenv()

if TYPE_CHECKING:
    from PIL.ImageFile import ImageFile


# Used to ensure that EventBus works across Gradio and the StateMachine.
def run_with_context(func: Callable) -> Callable:
    ctx = contextvars.copy_context()

    def wrapper(*args, **kwargs) -> Any:
        return ctx.run(func, *args, **kwargs)

    return wrapper


def get_image() -> ImageFile:
    graph = DotGraphMachine(machine.value).get_graph()
    return Image.open(BytesIO(graph.create_png()))


def create_statemachine() -> None:
    # Creates GoapMachine from the config.yaml in current directory
    cwd_path = Path.cwd()
    # config_path = cwd_path.joinpath(
    #     Path("examples/goap_machine/lifehunter_config.yaml")
    # )
    config_path = cwd_path.joinpath(Path("examples/goap_machine/config.yaml"))
    try:
        machine.value = GoapMachine.from_config_file(config_path)
    except Exception as e:
        raise gr.Error(str(e)) from e


# tuple[str, list[tuple[str, str]], dict[str, Any]]
def on_message(
    message: str, chat_history: list[tuple[str, str]]
) -> tuple[str, list[tuple[str, str]], dict[str, Any]]:
    if machine.value.current_state_value == "start":
        machine.value.start_machine()
        bot_message = "\n".join(machine.value.outputs_to_user)
        machine.value.outputs_to_user.clear()
    elif machine.value.current_state_value == "end":
        bot_message = "Thank you!"
    else:
        machine.value.send(
            "process_event",
            event_={"type": "user_input", "value": message},
        )
        bot_message = "\n".join(machine.value.outputs_to_user)
        machine.value.outputs_to_user.clear()
    chat_history.append((message, bot_message))
    return ("", chat_history, gr.update(value=get_image()))
    # return ("", chat_history)


autoscroll = """
    function Scrolldown() {
        let targetNode = document.querySelector('[aria-label="chatbot conversation"]')
        // Options for the observer (which mutations to observe)
        let config = { attributes: true, childList: true, subtree: true };

        // Callback function to execute when mutations are observed
        let callback = (mutationList, observer) => {
            targetNode.scrollTop = targetNode.scrollHeight;
        };

        // Create an observer instance linked to the callback function
        let observer = new MutationObserver(callback);

        // Start observing the target node for configured mutations
        observer.observe(targetNode, config);
    }
"""

with gr.Blocks(js=autoscroll) as demo:
    # Creates the statemachine before the gradio launches.
    machine = gr.State(value=None)
    create_statemachine()
    label = gr.Markdown("# GOAP State Machine Example")
    current_state = gr.Image(label="States", visible=True)
    with gr.Column():
        chatbot = gr.Chatbot()
        msg = gr.Textbox(
            label="Chat Input",
            placeholder="Type message here...",
        )
    msg.submit(
        run_with_context(on_message), [msg, chatbot], [msg, chatbot, current_state]
    )
# Launches the gradio app (the state machine is now in start_machine mode).
demo.launch()

machine.value.destroy_all_threads_and_rules()
