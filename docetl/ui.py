from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Footer, Input, Button, Label, Static
from textual.screen import Screen


class QueryInput(Static):
    """A widget for query input."""

    def compose(self) -> ComposeResult:
        yield Label("Data Path:")
        yield Input(placeholder="Enter the path to your data", id="data_path_input")
        yield Label("Query Description:")
        yield Input(
            placeholder="Describe your query in high-level terms", id="query_desc_input"
        )
        yield Button("Optimize Query", id="optimize_button", variant="primary")


class QueryResult(Static):
    """A widget to display query optimization results."""

    def compose(self) -> ComposeResult:
        yield Static("Results will appear here.", id="result_area")


class QueryOptimizer(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield ScrollableContainer(QueryInput(), QueryResult())
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "optimize_button":
            self.optimize_query()

    def optimize_query(self) -> None:
        data_path = self.query_one("#data_path_input").value
        query_desc = self.query_one("#query_desc_input").value

        if not data_path or not query_desc:
            self.query_one("#result_area").update("Please fill in both fields.")
            return

        # Simulated query optimization result
        result = f"""
Optimizing query...

Data path: {data_path}
Query description: {query_desc}

Optimized query plan:
1. Load data from specified path
2. Apply filters based on query description
3. Perform necessary joins
4. Aggregate results
5. Sort output
        """
        self.query_one("#result_area").update(result)


class QueryOptimizerApp(App):
    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def on_mount(self) -> None:
        self.push_screen(QueryOptimizer())

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark


if __name__ == "__main__":
    app = QueryOptimizerApp()
    app.run()
