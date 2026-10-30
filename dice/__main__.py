"""Run `python -m dice` to choose a scenario, optimise it and compare it with GAMS."""
from dice.model import display_scenarios
from dice.compare import run_scenario

if __name__ == "__main__":
    scenario = display_scenarios()
    run_scenario(scenario, compare=True, show=True)
