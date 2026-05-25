"""Local CLI for Estatia.

Thin client over the same compiled graph the API uses. Handles two
turn-shapes:

- Fresh turn: ``graph.invoke({"messages": [HumanMessage(...)]})``.
- Resume after a pending ``interrupt()``: ``graph.invoke(Command(resume=...))``.

Pending interrupts are detected via ``graph.get_state(config).interrupts``
on the per-thread checkpoint, so the CLI doesn't need to track the
pause/resume state itself.
"""

import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.graph.graph import build_graph, make_memory_checkpointer


EXIT_COMMANDS = {"salir", "quit", "exit"}


def _print_assistant_turn(graph, config, state: dict) -> None:
    """Print the pending interrupt question, or the latest AIMessage."""
    snapshot = graph.get_state(config)
    if snapshot.interrupts:
        question = snapshot.interrupts[0].value.get("clarification_question", "")
        if question:
            print(f"\nEstatia: {question}")
            return

    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            print(f"\nEstatia: {message.content}")
            return

    print("\nEstatia: (sin respuesta)")


def main() -> None:
    load_dotenv()

    graph = build_graph(checkpointer=make_memory_checkpointer())
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Estatia CLI — escribe 'salir' para terminar.")
    print(f"(thread_id: {thread_id})")

    while True:
        try:
            user_input = input("\nUsuario: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEstatia: ¡Hasta luego!")
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("\nEstatia: ¡Hasta luego!")
            break

        snapshot = graph.get_state(config)
        pending = bool(snapshot.interrupts) if snapshot else False

        try:
            if pending:
                state = graph.invoke(Command(resume=user_input), config=config)
            else:
                state = graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]}, config=config
                )
        except KeyboardInterrupt:
            print("\nEstatia: interrumpido. ¡Hasta luego!")
            break
        except Exception as exc:
            print(f"\nEstatia: error durante la ejecución del grafo: {exc}")
            continue

        _print_assistant_turn(graph, config, state)


if __name__ == "__main__":
    main()
