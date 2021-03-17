from asciimatics.screen import Screen

from momba import engine

from . import model


def game(screen):
    explorer = engine.Explorer.new_discrete_time(model.network)

    (state,) = explorer.initial_states

    while True:
        player = model.Cell(
            state.global_env["pos_x"].as_int, state.global_env["pos_y"].as_int
        )

        for y in range(model.track.height):
            for x in range(model.track.width):
                cell = model.Cell(x, y)
                if cell in model.track.obstacles:
                    screen.print_at("█", x, y, Screen.COLOUR_RED)
                elif cell == player:
                    screen.print_at("x", x, y, Screen.COLOUR_BLUE)
                else:
                    screen.print_at(" ", x, y)

        screen.refresh()

        if not state.transitions:
            screen.clear()
            if state.get_local_env(model.environment)["is_finished"].as_bool:
                screen.centre("Congratulations, you won! 🏆", screen.height // 2)
            if state.get_local_env(model.environment)["has_crashed"].as_bool:
                screen.centre("You crashed! 💥", screen.height // 2)
            screen.refresh()
            screen.wait_for_input(300)
            return

        screen.wait_for_input(300)
        event = screen.get_event()
        assert event is not None

        transitions = {
            transition.action.action_type.label: transition
            for transition in state.transitions
            if transition.action is not None
        }

        if event.key_code in {ord("Q"), ord("q")}:
            return

        if event.key_code == Screen.KEY_UP:
            transition = transitions["left"]
        elif event.key_code == Screen.KEY_DOWN:
            transition = transitions["right"]
        else:
            transition = transitions["stay"]

        state = transition.destinations.pick().state


Screen.wrapper(game)
