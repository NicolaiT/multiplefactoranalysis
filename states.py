from FeatureCloud.app.engine.app import AppState, app_state


# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('terminal')  # We declare that 'terminal' state is accessible from the 'initial' state.

    def run(self):
        return 'terminal'  # This means we are done. If the coordinator transitions into the 'terminal' state, the whole computation will be shut down.
