import uuid

from dotenv import load_dotenv

from src.graph.graph import build_graph


EXIT_COMMANDS = {"salir", "quit", "exit"}


def _print_assistant_message(state: dict) -> None:
    history = state.get("chat_history") or []
    for message in reversed(history):
        if message.get("role") == "assistant":
            print(f"\nEstatia: {message.get('content', '')}")
            return

    fallback = state.get("clarification_question")
    if fallback:
        print(f"\nEstatia: {fallback}")
        return

    print("\nEstatia: (sin respuesta)")


def main() -> None:
    load_dotenv()

    graph = build_graph()
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

        try:
            state = graph.invoke({"user_query": user_input}, config=config)
        except KeyboardInterrupt:
            print("\nEstatia: interrumpido. ¡Hasta luego!")
            break
        except Exception as exc:
            print(f"\nEstatia: error durante la ejecución del grafo: {exc}")
            continue

        _print_assistant_message(state)


if __name__ == "__main__":
    main()
