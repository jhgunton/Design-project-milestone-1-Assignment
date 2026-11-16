"""Build and explore the design project's ADC model."""

from modelingTools import ChemicalSpecies, CompiledModel

ASSESSMENT_DAY = 63.0
REFERENCE_DOSE_DAYS = (0.0, 21.0, 42.0)
DEFAULT_DOSE_MG_PER_KG = 1.0
MAX_DOSE_MG_PER_KG = 4.5
MIN_DOSE_INTERVAL_DAYS = 0.5
MAX_DOSE_INTERVAL_DAYS = (ASSESSMENT_DAY - 0.5) / 2


class ModelWithoutDilution(CompiledModel):
    """Omit growth dilution from all per-cell species in a compiled network."""

    def rhs(self, t, state):
        derivatives = super().rhs(t, state)
        for i, name in enumerate(self.species_names):
            if self.species(name).units == '#/cell':
                derivatives[i] += self.parameters['mu'] * state[i]
        return derivatives

    def rate_equations(self):
        return {
            name: equation._replace(terms=[term for term in equation.terms if term.kind != 'dilution'])
            for name, equation in super().rate_equations().items()
        }

    def moieties(self):
        moieties = super().moieties()
        for moiety in moieties:
            if any(weight and self.species(name).units == '#/cell'
                   for name, weight in moiety.coefficients.items()):
                moiety.changed_by.append('implied growth source from omitting dilution')
        return moieties


def build_model(network, parameters, time_units='min'):
    """Turn the project's reaction network and parameters into a runnable model.

    Use one irreversible ``N -> 2 N`` reaction with ``forward_rate_constant='mu'``
    for each replicated compartment with molecular species. Cell-loss reactions leave the
    surviving cells' per-cell amounts unchanged. Direct rate terms on cell counts
    and other growth laws are not supported by this function.

    The returned model supports Simulation, observables, accumulators, impulses,
    plotting, and equation inspection. Parameters retain their declared units.
    Cell division leaves mean molecular amounts per living cell unchanged.
    State variables and their derivatives use the species' declared units.
    """
    for compartment in network.replicated_compartments:
        if not any(isinstance(species, ChemicalSpecies) and species.compartment is compartment
                   for species in network.species.values()):
            continue
        count = compartment.count_variable
        growth = []
        for reaction in network.reactions:
            change = reaction.net_stoichiometry.get(count, 0)
            if change > 0:
                growth.append(reaction)
        if len(growth) != 1 or not (
            growth[0].reactants == {count: 1}
            and growth[0].products == {count: 2}
            and growth[0].forward_rate_constant == 'mu'
            and growth[0].rate_function is None
            and not growth[0].is_reversible
        ):
            raise ValueError(f"Use one exponential-growth reaction '{count} -> 2 {count}' with rate constant 'mu'.")
        if any(term.species == count for term in network.direct_terms):
            raise ValueError('Use reactions, rather than direct rate terms, to describe cell growth and death.')
    return ModelWithoutDilution(network, parameters, time_units)


# Preserve the entry point used by earlier exploration notebooks.
compile_without_dilution = build_model


def simulate_dosing_regimen(model, initial_values, dose_mg_per_kg, second_day, third_day, *, method='LSODA', atol=None):
    """Run three equal doses through the student's model and report day-63 results.

    The first dose is on day zero. Patient mass, molecular weights, dose limit,
    and assessment time are the assignment givens. Blood-payload AUC is integrated
    from the sampled concentration curve, without requiring a student accumulator.
    Neither the model nor the initial-values dictionary is changed.
    """
    import numpy as np
    from modelingTools import AVOGADRO, Impulse, Simulation

    if model is None or not initial_values:
        raise ValueError('Complete your model and untreated initial_values before launching the explorer.')
    if model.time_units != 'min':
        raise ValueError('Build your model with minutes as the simulation time unit.')
    if not np.isfinite(dose_mg_per_kg) or not 0 <= dose_mg_per_kg <= MAX_DOSE_MG_PER_KG:
        raise ValueError('Choose a dose between 0 and 4.5 mg/kg.')
    if not 0 < second_day < third_day < ASSESSMENT_DAY:
        raise ValueError(f'Choose dose days in order: 0 < second dose < third dose < {ASSESSMENT_DAY:g}.')
    required_initials = {'M_s', 'D_s', 'M_i', 'D_i', 'N'}
    if not required_initials.issubset(initial_values) or initial_values['N'] <= 0:
        raise ValueError('Define the four untreated receptor amounts and a positive N in initial_values first.')
    if any(initial_values.get(name, 0) != 0 for name in ['A_b', 'P_b', 'C_s', 'A_i', 'P_i', 'P_c']):
        raise ValueError('Start from untreated initial conditions, with no ADC or payload present.')
    for name, units in [('A_b', 'M'), ('P_b', 'M'), ('P_c', '#/cell'), ('N', 'cells')]:
        if model.species(name).units != units:
            raise ValueError(f'Use the assignment state name and units for {name}: {units}.')

    cell = model.species('P_c').compartment
    if cell is None or cell.count_variable != 'N' or not cell.volume_per_unit:
        raise ValueError('Give the cell compartment count_variable="N" and volume_per_unit in liters.')

    dose_days = [0.0, float(second_day), float(third_day)]
    edges = dose_days + [ASSESSMENT_DAY]
    grid = [np.array([0.0])]
    for start, end in zip(edges[:-1], edges[1:]):
        fine_end = min(start + 1, end)
        grid.append(np.linspace(start, fine_end, 301)[1:])
        if end > fine_end:
            grid.append(np.linspace(fine_end, end, int(20 * (end - fine_end)) + 2)[1:])

    dose_mol = dose_mg_per_kg * 70 * 1e-3 / 156000
    result = Simulation(
        'Three equal doses', np.concatenate(grid) * 1440, model,
        initial_values=dict(initial_values),
        impulses=[Impulse(day * 1440, 'A_b', dose_mol, 'mol') for day in dose_days],
        rtol=1e-9, atol=atol, method=method,
    ).simulate()
    days = result.in_days()
    adc = result['A_b'] * 156000 * 1e3
    payload = result['P_b'] * 1000 * 1e6
    cytoplasmic = result['P_c'] / (AVOGADRO * cell.volume_per_unit) * 1e6
    fraction = result['N'] / initial_values['N']
    if not all(np.all(np.isfinite(values)) for values in [adc, payload, cytoplasmic, fraction]):
        raise ValueError('The simulation returned nonfinite values. Check your model and initial conditions.')
    if np.any(fraction <= 0):
        raise ValueError('The simulation returned nonpositive cell counts. Check your model and solver tolerances.')

    return {
        'simulation': result, 'days': days, 'dose_days': dose_days,
        'dose_mg_per_kg': float(dose_mg_per_kg), 'total_dose_mg_per_kg': 3 * dose_mg_per_kg,
        'adc_ug_per_mL': adc, 'payload_ng_per_mL': payload,
        'cytoplasmic_uM': cytoplasmic, 'cell_fraction': fraction,
        'assessment_day': ASSESSMENT_DAY, 'final_fraction': float(fraction[-1]),
        'reduction_percent': float(100 * (1 - fraction[-1])),
        'target_met': bool(fraction[-1] <= 0.05),
        'payload_peak_ng_per_mL': float(np.max(payload)),
        'payload_auc_ug_h_per_mL': float(np.trapezoid(payload, days) * 24 / 1000),
    }


class DosingExperiment:
    """Run parameter revisions through a copy of the student's model.

    Values retain the parameter table's units. Every variant begins at
    its own untreated receptor steady state and the same initial tumor size.
    The notebook's parameters, model, and initial dictionary are never edited.
    """

    def __init__(self, model, initial_values):
        if model is None or not initial_values:
            raise ValueError('Complete your model and untreated initial_values before launching the explorer.')
        required = {'M_s', 'D_s', 'M_i', 'D_i', 'N'}
        if not required.issubset(initial_values):
            raise ValueError('Define all four untreated receptor amounts and N in initial_values first.')
        if model.time_units != 'min':
            raise ValueError('Build your model with minutes as the simulation time unit.')
        if any(initial_values.get(name, 0) != 0 for name in ['A_b', 'P_b', 'C_s', 'A_i', 'P_i', 'P_c']):
            raise ValueError('Use untreated initial_values, with no ADC or payload present.')
        self.model = model
        self.initial_values = dict(initial_values)
        self.defaults = dict(model.parameters)

    def configuration(self):
        return {
            'parameters': [
                {'name': name, 'value': value,
                 'units': str(self.model.parameters.units(name)),
                 'description': self.model.parameters.parameter(name).description}
                for name, value in self.defaults.items()
            ],
            'dose': DEFAULT_DOSE_MG_PER_KG, 'dose_interval': REFERENCE_DOSE_DAYS[1],
            'minimum_interval': MIN_DOSE_INTERVAL_DAYS, 'maximum_interval': MAX_DOSE_INTERVAL_DAYS,
            'assessment_day': ASSESSMENT_DAY,
            'maximum_dose': MAX_DOSE_MG_PER_KG, 'orders': 1.0, 'points': 5,
        }

    def _untreated_initials(self, model):
        import numpy as np
        from scipy.optimize import least_squares

        names = ['M_s', 'D_s', 'M_i', 'D_i']
        state = dict(self.initial_values)
        derivative = model.derivative_function()

        # Relative rates (per minute) weigh every receptor pool equally, however large or small.
        def residual(log_amounts):
            amounts = np.exp(log_amounts)
            rates = derivative(0, {**state, **dict(zip(names, amounts))})
            return np.array([rates[name] for name in names]) / amounts

        start = np.log([max(state[name], 1e-12) for name in names])
        steady = least_squares(residual, start, ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
        if not np.all(np.isfinite(steady.x)) or np.max(np.abs(residual(steady.x))) > 1e-10:
            raise ValueError('Could not find an untreated receptor steady state. Check synthesis, recycling, and degradation.')
        state.update(zip(names, np.exp(steady.x)))
        return state

    def _run(self, values, dose, second_day, third_day, label):
        import numpy as np

        parameters = self.model.parameters.with_changes(**values)
        model = type(self.model)(self.model.network, parameters, time_units='min')
        initials = self._untreated_initials(model)
        # Strong-killing sweeps need count tolerances below their smallest tumor.
        log_fraction_bound = -parameters.get('delta_max', 0) * ASSESSMENT_DAY * 1440
        count_bound = initials['N'] * np.exp(max(-600, log_fraction_bound))
        tolerance = min(1e-8, count_bound * 1e-10)
        # LSODA, the notebook's own solver, steps through the dose impulses that follow a
        # near-complete clearance of blood ADC; BDF fails there.
        report = simulate_dosing_regimen(model, initials, dose, second_day, third_day, atol={'N': tolerance})
        keys = ['days', 'adc_ug_per_mL', 'payload_ng_per_mL', 'cytoplasmic_uM', 'cell_fraction',
                'final_fraction', 'reduction_percent', 'payload_peak_ng_per_mL',
                'payload_auc_ug_h_per_mL', 'target_met']
        result = {key: report[key].tolist() if hasattr(report[key], 'tolist') else report[key] for key in keys}
        return {**result, 'label': label, 'parameters': values, 'initial_values': initials,
                'dose_mg_per_kg': dose, 'dose_days': [0, second_day, third_day]}

    def evaluate(self, request):
        import numpy as np

        values = {name: float(value) for name, value in request['parameters'].items()}
        if set(values) != set(self.defaults):
            raise ValueError('Supply a value for every parameter in the model.')
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError('Parameter values must be finite and nonnegative.')
        if values.get('K_P', 1) <= 0:
            raise ValueError('K_P must be positive.')
        dose, interval = (float(request[key]) for key in ['dose', 'dose_interval'])
        if not np.isfinite(dose) or not 0 <= dose <= MAX_DOSE_MG_PER_KG:
            raise ValueError(f'Choose a dose between 0 and {MAX_DOSE_MG_PER_KG:g} mg/kg.')
        if not np.isfinite(interval) or not MIN_DOSE_INTERVAL_DAYS <= interval <= MAX_DOSE_INTERVAL_DAYS:
            raise ValueError(f'Choose a dose interval from {MIN_DOSE_INTERVAL_DAYS:g} to '
                             f'{MAX_DOSE_INTERVAL_DAYS:g} days, keeping all three doses before day {ASSESSMENT_DAY:g}.')
        sweep_centers = {**values, 'dose': dose, 'dose_interval': interval}
        name = request.get('sweep')
        sweep_values = []
        range_limited = False
        if name:
            if name not in sweep_centers:
                raise ValueError('Choose a parameter from the sweep menu.')
            if sweep_centers[name] <= 0:
                raise ValueError('A logarithmic sweep needs a positive current parameter value.')
            orders = float(request['orders'])
            count = float(request['points'])
            if not np.isfinite(orders) or orders <= 0:
                raise ValueError('Orders of magnitude must be positive and finite.')
            if not np.isfinite(count) or not count.is_integer() or not 2 <= count <= 101:
                raise ValueError('Choose an integer number of sweep points from 2 to 101.')
            with np.errstate(over='ignore', under='ignore'):
                sweep_values = sweep_centers[name] * np.logspace(-orders, orders, int(count))
            if not np.all(np.isfinite(sweep_values)) or np.any(sweep_values <= 0):
                raise ValueError('This sweep range is too large. Reduce the orders of magnitude.')
            lower, upper = sweep_values[0], sweep_values[-1]
            if name == 'dose':
                upper = min(upper, MAX_DOSE_MG_PER_KG)
            elif name == 'dose_interval':
                lower = max(lower, MIN_DOSE_INTERVAL_DAYS)
                upper = min(upper, MAX_DOSE_INTERVAL_DAYS)
            range_limited = lower != sweep_values[0] or upper != sweep_values[-1]
            if range_limited:
                sweep_values = np.geomspace(lower, upper, int(count))

        reports = []
        if request.get('include_baseline', True):
            reports.append(self._run(values, dose, interval, 2 * interval, 'Current values'))
        for value in sweep_values:
            # The current-values curve already represents the midpoint of an odd sweep.
            if np.isclose(value, sweep_centers[name], rtol=1e-12, atol=0):
                continue
            changed = dict(values)
            trial_dose, trial_interval = dose, interval
            if name == 'dose':
                trial_dose = float(value)
                label = f'Dose size = {value:.4g} mg/kg'
            elif name == 'dose_interval':
                trial_interval = float(value)
                label = f'Dose interval = {value:.4g} days'
            else:
                changed[name] = float(value)
                label = f'{name} = {value:.4g} {self.model.parameters.units(name)}'
            try:
                reports.append(self._run(changed, trial_dose, trial_interval, 2 * trial_interval, label))
            except ValueError as error:
                raise ValueError(f'{label}: {error}') from error
        return {'reports': reports, 'sweep_values': list(sweep_values), 'range_limited': bool(range_limited)}


class DosingSession:
    """Keep notebook-defined rate functions in the kernel; send only JSON to the window."""

    def __init__(self, experiment):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import hmac
        import json
        import secrets
        import threading

        self.token = secrets.token_urlsafe(32)
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, status, value):
                body = json.dumps(value, allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def authorized(self):
                return hmac.compare_digest(self.headers.get('Authorization', ''), 'Bearer ' + token)

            def do_GET(self):
                if not self.authorized():
                    self.respond(403, {'error': 'This window is not connected to the notebook session.'})
                elif self.path == '/configuration':
                    self.respond(200, experiment.configuration())
                else:
                    self.respond(404, {'error': 'Unknown request.'})

            def do_POST(self):
                if not self.authorized():
                    self.respond(403, {'error': 'This window is not connected to the notebook session.'})
                    return
                if self.path == '/close':
                    try:
                        self.respond(200, {})
                    finally:
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                elif self.path == '/evaluate':
                    try:
                        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                        self.respond(200, experiment.evaluate(request))
                    except Exception as error:
                        self.respond(400, {'error': str(error)})
                else:
                    self.respond(404, {'error': 'Unknown request.'})

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        def serve():
            try:
                self.server.serve_forever()
            finally:
                self.server.server_close()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class DosingDemo:
    """A notebook launch button; no desktop window opens during Run All or builds."""

    def __init__(self, model_inputs):
        import ipywidgets as widgets

        self.model_inputs = model_inputs
        self.process = None
        self.session = None
        self.log = None
        self.launch_button = widgets.Button(description='Launch model explorer', button_style='info',
                                            layout=widgets.Layout(width='230px'))
        self.message = widgets.HTML('<p>Complete your model and untreated initial_values, then launch the separate window.</p>')
        self.widget = widgets.VBox([self.launch_button, self.message])
        self.launch_button.on_click(self.launch)

    def launch(self, _button=None):
        from html import escape
        from pathlib import Path
        import subprocess
        import sys
        import tempfile

        if self.process is not None and self.process.poll() is None:
            self.message.value = '<p>The explorer is already open. Close it and relaunch to load changes made in your notebook.</p>'
            return
        self.close()
        try:
            experiment = DosingExperiment(*self.model_inputs())
            self.session = DosingSession(experiment)
            self.log = tempfile.TemporaryFile(mode='w+t')
            script = Path(__file__).with_name('design_project_demo.py')
            self.process = subprocess.Popen(
                [sys.executable, str(script), self.session.url, self.session.token],
                stdout=self.log, stderr=self.log,
            )
            try:
                self.process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.message.value = ('<p>Explorer launched in its own window. Keep this notebook kernel running. '
                                      'Close and relaunch the window after changing your notebook model.</p>')
                return
            self.log.seek(0)
            raise RuntimeError('The explorer could not open: ' + (self.log.read().strip() or 'window closed during startup.'))
        except Exception as error:
            self.close()
            self.message.value = '<p>' + escape(str(error)) + '</p>'

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.log is not None:
            self.log.close()
            self.log = None


def show_dosing_demo(model_inputs):
    """Display a button that opens the native model explorer."""
    from IPython.display import display

    demo = DosingDemo(model_inputs)
    bundle = demo.widget._repr_mimebundle_()
    bundle['text/plain'] = 'Launch model explorer: open the notebook to launch the separate window.'
    display(bundle, raw=True)
    return demo
