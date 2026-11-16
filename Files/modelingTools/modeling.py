"""Helper module for the 20.320 collaborative modeling project.

The module follows the unit conventions of the receptor trafficking lecture:

* Species that live on or inside cells are declared in **molecules per cell** (`'#/cell'`).
* Species in a bulk fluid such as blood are declared in **molar** (`'M'`).
* The number of cells is itself a state variable with units `'cells'`.

Every species belongs to a compartment. A *bulk* compartment has a fixed volume. A
*replicated* compartment (the cell) creates its own count variable and has, optionally, a volume per cell.
Rates are written in the units the lecture uses: a reaction that touches anything on a cell has
a rate in molecules per cell per time, and the model multiplies by the cell count N and divides
by Avogadro's number and the fluid volume wherever a molar species is involved. The generated
rate equations print those factors explicitly, in the same form as the lecture's ligand
balance, N_Av V d[L]/dt = - N k_f [R_s][L] + ...

Workflow: declare compartments, species, reactions, observables and accumulators on a
``KineticModel``; declare parameters with units in a ``Parameters`` table; call
``kinetic_model.compile(parameters, time_units)`` to validate everything at once and obtain a
``CompiledModel``; run it with ``Simulation``. The compiled model exposes its stoichiometry
matrix, the moieties that follow from it with what changes each one, its rate equations in the
lecture's form, and a derivative function for checks.

Every species is integrated in its declared units. Transfers between compartments use the
cell counts, volumes, and Avogadro's number to balance molecular amounts. Growth dilutes
per-cell values automatically; cell-loss reactions leave the surviving cells' per-cell
values unchanged. Population totals are calculated for reporting.

Nothing in this module fails silently. Undeclared species, unknown parameters, units that do
not work out, and solver failures all raise ``ValueError`` with a message that says what to fix.
"""

import re
from collections import namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

AVOGADRO = 6.02214076e23

TIME_UNITS_IN_SECONDS = {'sec': 1.0, 'min': 60.0, 'hr': 3600.0, 'day': 86400.0}

TIME_TOKEN_ALIASES = {
    's': 'sec', 'sec': 'sec', 'second': 'sec', 'seconds': 'sec',
    'min': 'min', 'minute': 'min', 'minutes': 'min',
    'h': 'hr', 'hr': 'hr', 'hour': 'hr', 'hours': 'hr',
    'd': 'day', 'day': 'day', 'days': 'day',
}

SPECIES_UNITS = ('M', '#/cell')

IMPULSE_UNITS = ('mol', 'M', 'molecules', '#/cell', 'cells', 'raw')


# ===== Units ===================================================================================

BASE_ALIASES = {
    '#': '#', 'molecule': '#', 'molecules': '#',
    'cell': 'cell', 'cells': 'cell',
    'M': 'M', 'L': 'L', 'mol': 'mol', 'g': 'g',
}

_UNIT_TOKEN = re.compile(r'\(([^()]+)\)(?:\^(-?\d+))?|([A-Za-z#]+)(?:\^(-?\d+))?|(/)|(\s+)')


class Unit:
    """Units over a closed vocabulary: M, #/cell (molecules per cell), cells, L, mol, g, time.

    ``Unit('M^-1 min^-1')``, ``Unit('#/cell/min')``, ``Unit('(#/cell)^-1 min^-1')``,
    ``Unit('M min')``, ``Unit('mol')``, ``Unit('1')`` (dimensionless). Products, powers and
    equality are supported. The time unit name (sec, min, hr, day) is carried along so
    mismatches can be reported.
    """

    def __init__(self, text=None, exponents=None, time_unit=None):
        if text is not None:
            exponents, time_unit = self._parse(text)
        self.exponents = {base: power for base, power in (exponents or {}).items() if power != 0}
        self.time_unit = time_unit if self.exponents.get('time') else None
        self.text = text

    @classmethod
    def _parse(cls, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A units string is required (for example 'min^-1' or 'M^-1 min^-1').")
        if text.strip() == '1':
            return {}, None
        exponents, time_unit = {}, None
        position, invert = 0, False
        for match in _UNIT_TOKEN.finditer(text):
            if match.start() != position:
                raise ValueError(f"Cannot read units '{text}' at '{text[position:]}'.")
            position = match.end()
            group, group_power, base, base_power, slash, _space = match.groups()
            if slash:
                invert = True
                continue
            if group is None and base is None:
                continue
            sign = -1 if invert else 1
            invert = False
            if group is not None:
                inner, inner_time = cls._parse(group)
                power = int(group_power) if group_power else 1
                for key, value in inner.items():
                    exponents[key] = exponents.get(key, 0) + sign * power * value
                if inner_time:
                    if time_unit and inner_time != time_unit:
                        raise ValueError(f"Units '{text}' name two time units.")
                    time_unit = inner_time
                continue
            power = int(base_power) if base_power else 1
            if base in TIME_TOKEN_ALIASES:
                canonical = TIME_TOKEN_ALIASES[base]
                if time_unit and canonical != time_unit:
                    raise ValueError(f"Units '{text}' name two time units ({time_unit} and {canonical}).")
                time_unit = canonical
                exponents['time'] = exponents.get('time', 0) + sign * power
            elif base in BASE_ALIASES:
                key = BASE_ALIASES[base]
                exponents[key] = exponents.get(key, 0) + sign * power
            else:
                raise ValueError(
                    f"Unknown unit '{base}' in '{text}'. Use M, #/cell (or molecules/cell), cells, L, mol, g, "
                    "and a time unit (sec, min, hr, day)."
                )
        if position != len(text):
            raise ValueError(f"Cannot read units '{text}' at '{text[position:]}'.")
        return exponents, time_unit

    def __mul__(self, other):
        exponents = dict(self.exponents)
        for key, value in other.exponents.items():
            exponents[key] = exponents.get(key, 0) + value
        time_unit = self.time_unit or other.time_unit
        if self.time_unit and other.time_unit and self.time_unit != other.time_unit:
            raise ValueError(f"Cannot combine time units {self.time_unit} and {other.time_unit}.")
        return Unit(exponents=exponents, time_unit=time_unit)

    def __pow__(self, power):
        return Unit(exponents={key: value * power for key, value in self.exponents.items()},
                    time_unit=self.time_unit)

    def __truediv__(self, other):
        return self * other ** -1

    def __eq__(self, other):
        return isinstance(other, Unit) and self.exponents == other.exponents

    def __hash__(self):
        return hash(tuple(sorted(self.exponents.items())))

    def __str__(self):
        if not self.exponents:
            return '1'
        exponents = dict(self.exponents)
        parts = []
        if exponents.get('M'):
            power = exponents.pop('M')
            parts.append('M' if power == 1 else f'M^{power}')
        # Molecules per cell is written as the single symbol '#/cell' whenever # and cell
        # appear with opposite powers, so '(#/cell)^-1 min^-1' reads back as written.
        per_cell = exponents.get('#', 0)
        if per_cell and exponents.get('cell', 0) == -per_cell:
            parts.append('#/cell' if per_cell == 1 else f'(#/cell)^{per_cell}')
            del exponents['#'], exponents['cell']
        for key in ['#', 'cell', 'mol', 'g', 'L', 'time']:
            power = exponents.get(key)
            if not power:
                continue
            name = self.time_unit if key == 'time' else ('cells' if key == 'cell' else key)
            parts.append(name if power == 1 else f'{name}^{power}')
        return ' '.join(parts)

    def __repr__(self):
        return f"Unit('{self}')"

    def per_time(self):
        return self.exponents.get('time', 0)


def unit_of_species(units):
    return {'M': Unit('M'), '#/cell': Unit('#/cell'), 'cells': Unit('cells')}[units]


def time_unit_in(units_string):
    """Return the canonical time unit named in a units string, or None if it names none."""
    return Unit(units_string).time_unit


def check_time_units(units, time_units, what):
    """Raise if a Unit names a time unit different from the simulation's."""
    if units.time_unit is not None and units.time_unit != time_units:
        raise ValueError(
            f"{what} has units '{units}', but the simulation runs in '{time_units}'. "
            f"Convert the value to {time_units} or change time_units."
        )


def convert_rate_constant(value, from_units, to_units, volume):
    """Convert between molar and per-cell forms using a reference volume in liters.

    A molar quantity becomes a per-cell quantity by multiplying by Avogadro's number times the
    volume, so 'M' -> '#/cell' multiplies by N_Av V, 'M^-1 min^-1' -> '(#/cell)^-1 min^-1'
    divides by it, and higher powers follow. The factor used is printed.
    """
    source, target = Unit(from_units), Unit(to_units)
    if source.time_unit != target.time_unit or source.per_time() != target.per_time():
        raise ValueError(
            f"convert_rate_constant does not change time units ('{from_units}' to '{to_units}'); "
            "convert the time unit separately."
        )
    difference = (source / target).exponents
    molar_power = difference.get('M', 0)
    expected = {'M': molar_power, '#': -molar_power, 'cell': molar_power}
    if molar_power == 0 or {k: v for k, v in difference.items() if v} != {k: v for k, v in expected.items() if v}:
        raise ValueError(
            f"No conversion from '{from_units}' to '{to_units}'. The two must differ only by "
            "replacing M with #/cell (or the reverse), to any power."
        )
    molecules_per_cell_per_molar = AVOGADRO * volume
    factor = molecules_per_cell_per_molar ** molar_power
    converted = value * factor
    print(f"convert_rate_constant: {value:.4g} {from_units} × {factor:.4g} = {converted:.4g} {to_units} "
          f"(N_Av × {volume:.4g} L = {molecules_per_cell_per_molar:.4g} molecules per cell per molar)")
    return converted


# ===== Parameters ==============================================================================

@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    units: Unit
    description: str = ''


class Parameters(Mapping):
    """Named parameters with values, units and descriptions.

    ``k['k_on']`` and ``k.k_on`` return the value; ``k.parameter('k_on')`` returns the
    ``Parameter``; ``k.add(name, value, units, description)`` declares one;
    ``k.with_changes(k_on=...)`` returns a new table for sweeps.
    """

    def __init__(self, parameters=()):
        object.__setattr__(self, '_parameters', {})
        for parameter in parameters:
            self._parameters[parameter.name] = parameter

    def add(self, name, value, units, description=''):
        if name in self._parameters:
            raise ValueError(f"Parameter '{name}' is already declared; use with_changes to alter it.")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Parameter name '{name}' must be a valid identifier (letters, digits, underscores).")
        self._parameters[name] = Parameter(name, float(value), Unit(units), description)
        return self._parameters[name]

    def parameter(self, name):
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' is not declared. Declare it with add(name, value, units).")
        return self._parameters[name]

    def __getitem__(self, name):
        return self.parameter(name).value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._parameters[name].value
        except KeyError:
            raise AttributeError(f"Parameter '{name}' is not declared.") from None

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self):
        return len(self._parameters)

    def __contains__(self, name):
        return name in self._parameters

    def units(self, name):
        return self.parameter(name).units

    def with_changes(self, **changes):
        """A new table with the given values replaced; units and descriptions are kept."""
        table = Parameters(self._parameters.values())
        for name, value in changes.items():
            old = table.parameter(name)
            table._parameters[name] = Parameter(name, float(value), old.units, old.description)
        return table

    def __str__(self):
        if not self._parameters:
            return 'No parameters declared'
        width = max(len(name) for name in self._parameters)
        lines = [f"{'Parameter':<{width}}  {'Value':>11}  {'Units':<20}  Description", '-' * (width + 60)]
        for parameter in self._parameters.values():
            value = parameter.value
            value_text = f'{value:.4g}' if 1e-3 <= abs(value) <= 1e4 or value == 0 else f'{value:.3e}'
            lines.append(f"{parameter.name:<{width}}  {value_text:>11}  {str(parameter.units):<20}  {parameter.description}")
        return '\n'.join(lines)

    def __repr__(self):
        return f"Parameters({list(self._parameters.values())!r})"


# ===== Declarations ============================================================================

@dataclass
class Compartment:
    """A bulk compartment (fixed volume) or a replicated compartment (counted by a state variable).

    volume : liters of a bulk compartment such as blood.
    count_variable : name of the population-count state created when this compartment is added.
    count_units : units of that count, defaulting to 'cells' for a replicated compartment.
    count_description : optional label for the count state.
    volume_per_unit : liters of one unit of a replicated compartment (one cell), used to express a
        per-cell quantity as a concentration.
    """
    name: str
    volume: float = None
    count_variable: str = None
    volume_per_unit: float = None
    count_units: str = None
    count_description: str = None

    def __post_init__(self):
        if (self.volume is None) == (self.count_variable is None):
            raise ValueError(
                f"Compartment '{self.name}' must have exactly one of volume (bulk) or count_variable (replicated)."
            )
        if self.volume is not None and (not np.isfinite(self.volume) or self.volume <= 0):
            raise ValueError(f"Compartment '{self.name}' volume must be positive.")
        if self.volume is not None and self.volume_per_unit is not None:
            raise ValueError(f"Compartment '{self.name}' is bulk; volume_per_unit applies only to replicated compartments.")
        if self.volume is not None and (self.count_units is not None or self.count_description is not None):
            raise ValueError(f"Compartment '{self.name}' is bulk; count_units and count_description need count_variable.")
        if self.count_variable is not None:
            if self.count_units is None:
                self.count_units = 'cells'
            if Unit(self.count_units) != Unit('cells'):
                raise ValueError(f"Compartment '{self.name}' count_units must be compatible with 'cells'.")
            self.count_units = str(Unit(self.count_units))
        if self.volume_per_unit is not None and (
                not np.isfinite(self.volume_per_unit) or self.volume_per_unit <= 0):
            raise ValueError(f"Compartment '{self.name}' volume_per_unit must be positive.")

    @property
    def is_replicated(self):
        return self.count_variable is not None


@dataclass
class StateVariable:
    name: str
    description: str
    units: str


@dataclass
class PopulationCount(StateVariable):
    population: Compartment

    def __str__(self):
        return f'{self.description} ({self.units}; counts {self.population.name})'


@dataclass
class ChemicalSpecies(StateVariable):
    compartment: Compartment = None

    def __str__(self):
        where = f" in {self.compartment.name}" if self.compartment else ''
        return f"{self.description} ({self.units}){where}"


@dataclass
class DirectTerm:
    species: str
    function: object
    expression: str
    units: Unit
    description: str = ''


@dataclass
class Observable:
    name: str
    function: object
    units: str
    description: str = ''


@dataclass
class Accumulator:
    name: str
    rate_function: object
    units: Unit
    description: str = ''


def _integer_coefficient(coefficient, species, reaction_name):
    """Accept positive integers, including exactly integer-valued floats and numeric text."""
    original = coefficient
    if isinstance(coefficient, str):
        try:
            number = Fraction(coefficient)
            coefficient = number.numerator if number.denominator == 1 else None
        except (ValueError, ZeroDivisionError):
            coefficient = None
    is_integer = isinstance(coefficient, (int, np.integer)) and not isinstance(coefficient, (bool, np.bool_))
    is_integral_float = (isinstance(coefficient, (float, np.floating))
                         and np.isfinite(coefficient) and coefficient.is_integer())
    if not (is_integer or is_integral_float) or coefficient <= 0:
        raise ValueError(
            f"Reaction '{reaction_name}': coefficient for '{species}' must be a positive integer "
            f"(got {original!r})."
        )
    return int(coefficient)


def _parse_species_text(text, reaction_name):
    """Parse '2 M_s + A_b' into {'M_s': 2, 'A_b': 1}."""
    stoichiometry = {}
    for part in text.split('+'):
        tokens = part.strip().split()
        if not tokens:
            continue
        if len(tokens) == 2:
            coefficient, species = tokens
            coefficient = _integer_coefficient(coefficient, species, reaction_name)
        elif len(tokens) == 1:
            coefficient, species = 1, tokens[0]
        else:
            raise ValueError(f"Reaction '{reaction_name}': cannot parse '{part.strip()}' as "
                             "'<positive integer> <species>' or '<species>'.")
        stoichiometry[species] = stoichiometry.get(species, 0) + coefficient
    return stoichiometry


def _stoichiometry_from(species_input, what, reaction_name):
    if isinstance(species_input, dict):
        return {species: _integer_coefficient(coefficient, species, reaction_name)
                for species, coefficient in species_input.items()}
    if isinstance(species_input, str):
        return _parse_species_text(species_input, reaction_name)
    if species_input is None or (isinstance(species_input, (list, tuple)) and not species_input):
        return {}
    raise ValueError(f"Reaction '{reaction_name}': {what} must be a dict, a string like '2 M_s + A_b', or empty.")


class Reaction:
    """One reaction with mass-action rates generated from its stoichiometry, or custom rates.

    reactants, products : ``{'M_s': 2}`` or ``'2 M_s'``; empty for synthesis or loss.
        Integer-valued floats such as 2.0 are accepted and stored as integers.
    forward_rate_constant, reverse_rate_constant : parameter names; the rate is the constant times
        the product of the reactant (or product) values raised to their coefficients.
    rate_function : ``rate_function(s, k)`` returning the forward rate in the reaction's reference
        units: molecules per cell per time when any participant is per cell, molar per time when all
        are molar, cells per time for growth and death. ``s`` is a State, ``k`` the Parameters.
        Requires ``rate_expression``, the text printed in the rate equations, and ``rate_units``
        (checked for dimensions and time unit at compile time). Units inside the Python function
        are not inferred; convert parameter values as needed to return these rate_units.
        The function must be smooth in the state:
        write a threshold as ``smooth_step(...)`` rather than an ``if``, because the integrator's
        step control assumes a smooth right-hand side.
    reverse_rate_function, reverse_rate_expression : custom reverse rate, same rules.
    Cell-loss reactions leave the surviving cells' per-cell amounts unchanged. Redistribution
    or release of molecular contents must be described by explicit molecular source reactions.
    """

    def __init__(self, name, reactants, products, forward_rate_constant=None,
                 reverse_rate_constant=None, rate_function=None, rate_expression=None,
                 reverse_rate_function=None, reverse_rate_expression=None,
                 description='', rate_expression_latex=None,
                 reverse_rate_expression_latex=None, rate_units=None):
        self.name = name
        self.reactants = _stoichiometry_from(reactants, 'reactants', name)
        self.products = _stoichiometry_from(products, 'products', name)
        if not self.reactants and not self.products:
            raise ValueError(f"Reaction '{name}' has no reactants and no products.")
        if (forward_rate_constant is None) == (rate_function is None):
            raise ValueError(f"Reaction '{name}': give exactly one of forward_rate_constant or rate_function.")
        if rate_function is not None and not rate_expression:
            raise ValueError(
                f"Reaction '{name}': a rate_function needs a rate_expression string, the text printed in "
                "the rate equations (for example 'k_on × [A_b] × [M_s]')."
            )
        if reverse_rate_constant is not None and reverse_rate_function is not None:
            raise ValueError(f"Reaction '{name}': give at most one of reverse_rate_constant or reverse_rate_function.")
        if reverse_rate_function is not None and not reverse_rate_expression:
            raise ValueError(f"Reaction '{name}': a reverse_rate_function needs a reverse_rate_expression string.")
        if (rate_function is not None or reverse_rate_function is not None) and not rate_units:
            raise ValueError(f"Reaction '{name}': a custom forward or reverse rate function needs rate_units.")
        self.forward_rate_constant = forward_rate_constant
        self.reverse_rate_constant = reverse_rate_constant
        self.rate_function = rate_function
        self.rate_expression = rate_expression
        self.rate_expression_latex = rate_expression_latex
        self.rate_units = Unit(rate_units) if rate_units else None
        self.reverse_rate_function = reverse_rate_function
        self.reverse_rate_expression = reverse_rate_expression
        self.reverse_rate_expression_latex = reverse_rate_expression_latex
        self.description = description

    @property
    def is_reversible(self):
        return self.reverse_rate_constant is not None or self.reverse_rate_function is not None

    @property
    def participants(self):
        return list(dict.fromkeys(list(self.reactants) + list(self.products)))

    @property
    def net_stoichiometry(self):
        net = {}
        for species, coefficient in self.reactants.items():
            net[species] = net.get(species, 0) - coefficient
        for species, coefficient in self.products.items():
            net[species] = net.get(species, 0) + coefficient
        return {species: value for species, value in net.items() if value != 0}

    @property
    def is_source_or_sink(self):
        return not self.reactants or not self.products

    def parameter_names(self):
        return [name for name in (self.forward_rate_constant, self.reverse_rate_constant) if name is not None]

    def forward_rate(self, s, k):
        if self.rate_function is not None:
            return self.rate_function(s, k)
        return _mass_action(k[self.forward_rate_constant], self.reactants, s)

    def reverse_rate(self, s, k):
        if self.reverse_rate_function is not None:
            return self.reverse_rate_function(s, k)
        if self.reverse_rate_constant is not None:
            return _mass_action(k[self.reverse_rate_constant], self.products, s)
        return 0.0

    @staticmethod
    def _side_text(stoichiometry):
        if not stoichiometry:
            return '∅'
        return ' + '.join(species if coefficient == 1 else f'{coefficient} {species}'
                          for species, coefficient in stoichiometry.items())

    def __str__(self):
        arrow = '⇌' if self.is_reversible else '→'
        equation = f"{self._side_text(self.reactants)} {arrow} {self._side_text(self.products)}"
        forward = self.rate_expression if self.rate_function is not None else f"k = {self.forward_rate_constant}"
        if self.is_reversible:
            reverse = (self.reverse_rate_expression if self.reverse_rate_function is not None
                       else f"k = {self.reverse_rate_constant}")
            rate_info = f"(forward {forward}; reverse {reverse})"
        else:
            rate_info = f"({forward})"
        text = f"{self.name}: {equation} {rate_info}"
        if self.description:
            text += f" — {self.description}"
        return text


def _mass_action(rate_constant, stoichiometry, s):
    rate = rate_constant
    for species, coefficient in stoichiometry.items():
        rate = rate * (s[species] if coefficient == 1 else s[species] ** coefficient)
    return rate


def _report_replacement(kind, name):
    print(f"Warning: replaced the existing {kind} '{name}'. A model built from this network "
          "before the change still uses the old version; rebuild the model (for example, call "
          "build_model or compile again) so it uses the new one.")


class KineticModel:
    """Compartments, species, reactions, direct rate terms, observables, and accumulators.

    Declare everything here, then ``compile(parameters, time_units)`` to get a validated model.
    The species mapping contains molecular species and compartment-owned population counts.
    """

    def __init__(self):
        self.compartments = {}
        self.species = {}
        self.reactions = []
        self.direct_terms = []
        self.observables = {}
        self.accumulators = {}

    def add_compartment(self, compartment):
        """Register a compartment and create its count state when count_variable is supplied."""
        if compartment.name in self.compartments:
            raise ValueError(f"Compartment '{compartment.name}' is already defined.")
        if compartment.is_replicated:
            count = compartment.count_variable
            self._validate_state_name(count)
            description = compartment.count_description or f'Count of {compartment.name}'
            self.species[count] = PopulationCount(count, description, compartment.count_units, compartment)
        self.compartments[compartment.name] = compartment
        return compartment

    def _validate_state_name(self, name):
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"State name '{name}' must be a valid identifier (letters, digits, underscores).")
        if name in self.species or name in self.accumulators or name in self.observables:
            raise ValueError(f"'{name}' is already declared.")

    def add_species(self, name, description, units, compartment=None):
        self._validate_state_name(name)
        if units not in SPECIES_UNITS:
            raise ValueError(f"Species '{name}' has units '{units}'; use one of {SPECIES_UNITS}. "
                             "Declare population counts with Compartment(count_variable=...).")
        if isinstance(compartment, str):
            if compartment not in self.compartments:
                raise ValueError(
                    f"Species '{name}' names compartment '{compartment}', which is not defined. Call add_compartment first."
                )
            compartment = self.compartments[compartment]
        if compartment is None:
            raise ValueError(f"Species '{name}' ({units}) needs a compartment.")
        elif self.compartments.get(compartment.name) is not compartment:
            raise ValueError(f"Species '{name}' needs a compartment registered with add_compartment.")
        elif units == 'M' and compartment.is_replicated:
            raise ValueError(
                f"Species '{name}' is molar but '{compartment.name}' is a replicated compartment; "
                "use '#/cell' there or a bulk compartment for molar species."
            )
        elif units == '#/cell' and not compartment.is_replicated:
            raise ValueError(
                f"Species '{name}' is per cell but '{compartment.name}' is a bulk compartment; "
                "use 'M' there or a replicated compartment for per-cell species."
            )
        self.species[name] = ChemicalSpecies(name, description, units, compartment)

    def add_reaction(self, reaction):
        """Add a reaction; one with the same name as an existing reaction replaces it."""
        for species in reaction.participants:
            if species not in self.species:
                raise ValueError(
                    f"Reaction '{reaction.name}' uses species '{species}', which is not declared. "
                    "Declare molecular species with add_species and population counts with add_compartment "
                    "before adding reactions."
                )
        self.reference_of(reaction)  # validates the compartment rules now
        count_species = [name for name in reaction.participants
                         if isinstance(self.species[name], PopulationCount)]
        if count_species and reaction.is_reversible:
            raise ValueError(
                f"Reaction '{reaction.name}' changes a cell count and cannot be reversible; "
                "write growth and death as separate reactions."
            )
        existing_names = [existing.name for existing in self.reactions]
        if reaction.name in existing_names:
            self.reactions[existing_names.index(reaction.name)] = reaction
            _report_replacement('reaction', reaction.name)
        else:
            self.reactions.append(reaction)
        return reaction

    def add_rate_term(self, species, function, expression, units, description=''):
        if species not in self.species:
            raise ValueError(f"add_rate_term: species '{species}' is not declared.")
        if not expression:
            raise ValueError(f"add_rate_term for '{species}' needs an expression string to print in the rate equations.")
        self.direct_terms.append(DirectTerm(species, function, expression, Unit(units), description))

    def add_observable(self, name, function, units, description=''):
        """Register a function returning a scalar constant or one value per output time.

        An observable with the same name as an existing observable replaces it.
        """
        if name in self.species or name in self.accumulators:
            raise ValueError(f"'{name}' is already declared.")
        replacing = name in self.observables
        self.observables[name] = Observable(name, function, units, description)
        if replacing:
            _report_replacement('observable', name)

    def add_accumulator(self, name, rate_function, units, description=''):
        """Register an integrated quantity; one with an existing accumulator's name replaces it."""
        if name in self.species or name in self.observables:
            raise ValueError(f"'{name}' is already declared.")
        replacing = name in self.accumulators
        self.accumulators[name] = Accumulator(name, rate_function, Unit(units), description)
        if replacing:
            _report_replacement('accumulator', name)

    @property
    def replicated_compartments(self):
        return [c for c in self.compartments.values() if c.is_replicated]

    def reference_of(self, reaction):
        """('cells', None), ('per_cell', compartment) or ('bulk', compartment) for a reaction."""
        counts = [name for name in reaction.participants if isinstance(self.species[name], PopulationCount)]
        if counts:
            if len(counts) != len(reaction.participants):
                raise ValueError(
                    f"Reaction '{reaction.name}' mixes a cell count with molecular species. Write cell growth and "
                    "death as reactions on the count alone, and molecular release as separate source reactions."
                )
            return ('cells', None)
        replicated = {self.species[name].compartment.name for name in reaction.participants
                      if self.species[name].compartment.is_replicated}
        if len(replicated) > 1:
            raise ValueError(
                f"Reaction '{reaction.name}' spans two replicated compartments ({sorted(replicated)}); "
                "a per-cell rate has one cell type as its reference."
            )
        if replicated:
            return ('per_cell', self.compartments[replicated.pop()])
        return ('bulk', self.species[reaction.participants[0]].compartment)

    def compile(self, parameters, time_units='min'):
        """Validate the whole model against its parameters and return a CompiledModel."""
        return CompiledModel(self, parameters, time_units)

    def __str__(self):
        lines = [f"Kinetic model: {len(self.species)} species, {len(self.reactions)} reactions"]
        for compartment in self.compartments.values():
            if compartment.is_replicated:
                per_unit = f", {compartment.volume_per_unit:.4g} L each" if compartment.volume_per_unit else ''
                lines.append(f"  compartment {compartment.name}: counted by {compartment.count_variable}{per_unit}")
            else:
                lines.append(f"  compartment {compartment.name}: {compartment.volume:.4g} L")
        for species in self.species.values():
            lines.append(f"  {species.name}: {species}")
        for i, reaction in enumerate(self.reactions, 1):
            lines.append(f"  {i:2d}. {reaction}")
        for term in self.direct_terms:
            lines.append(f"  direct term on {term.species}: {term.expression} ({term.units})")
        for accumulator in self.accumulators.values():
            lines.append(f"  accumulator {accumulator.name} ({accumulator.units}): {accumulator.description}")
        for observable in self.observables.values():
            lines.append(f"  observable {observable.name} ({observable.units}): {observable.description}")
        return '\n'.join(lines)


# ===== State ===================================================================================

class State:
    """Species values in declared units, with the conversions rate functions need.

    ``s['P_c']`` or ``s.P_c``: the declared value (molecules per cell, molar, or cells).
    ``s.molar('P_c')``: a per-cell species as a concentration in one cell's volume; a molar
    species unchanged. ``s.per_cell('A_b', 'cell')``: the reverse. ``s.count('cell')`` and
    ``s.volume('blood')``: compartment size. ``s.observe('delta')``: a registered observable
    evaluated on this state. Values may be scalars (during integration) or arrays (results).
    """

    def __init__(self, values, model):
        self._values = values
        self._model = model

    def __getitem__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise KeyError(f"'{name}' is not a species or accumulator of this model.") from None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(f"'{name}' is not a species or accumulator of this model.") from None

    def __contains__(self, name):
        return name in self._values

    def keys(self):
        return self._values.keys()

    def count(self, compartment_name):
        compartment = self._model.compartment(compartment_name)
        if not compartment.is_replicated:
            raise ValueError(f"Compartment '{compartment_name}' is bulk; it has a volume, not a count.")
        return self._values[compartment.count_variable]

    def volume(self, compartment_name):
        compartment = self._model.compartment(compartment_name)
        if compartment.is_replicated:
            if compartment.volume_per_unit is None:
                raise ValueError(
                    f"Compartment '{compartment_name}' has no volume_per_unit; give one to express per-cell "
                    "quantities as concentrations."
                )
            return compartment.volume_per_unit
        return compartment.volume

    def molar(self, name):
        species = self._model.species(name)
        if isinstance(species, PopulationCount):
            raise ValueError(f"'{name}' is a cell count and has no concentration.")
        if species.compartment.is_replicated:
            return self._values[name] / (self.volume(species.compartment.name) * AVOGADRO)
        return self._values[name]

    def per_cell(self, name, compartment=None):
        species = self._model.species(name)
        if isinstance(species, PopulationCount):
            raise ValueError(f"'{name}' is a cell count and has no per-cell value.")
        if species.compartment.is_replicated:
            return self._values[name]
        if compartment is None:
            replicated = [c for c in self._model.network.replicated_compartments
                          if c.volume_per_unit is not None]
            if len(replicated) != 1:
                raise ValueError("per_cell needs a compartment name when there is not exactly one counted "
                                 "compartment with volume_per_unit.")
            compartment = replicated[0].name
        return self._values[name] * self.volume(compartment) * AVOGADRO

    def observe(self, name):
        observable = self._model.network.observables.get(name)
        if observable is None:
            raise KeyError(f"'{name}' is not a registered observable.")
        return observable.function(self, self._model.parameters)


# ===== Compiled model ==========================================================================

RateEquation = namedtuple('RateEquation', ['species', 'lhs_text', 'lhs_latex', 'terms'])
RateTerm = namedtuple('RateTerm', ['text', 'latex', 'kind'])


@dataclass
class Moiety:
    """A combination of species that the model's reactions shuffle but do not create or destroy.

    coefficients : species name -> integer weight (a dimer counts twice, a loaded antibody eight).
    text : the combination written out, e.g. 'M_s + 2 D_s + C_s + M_i + 2 D_i'.
    total_text : the combination in total molecules (or cells), with volume and cell-count factors.
    changed_by : the sources, sinks, direct terms, and death removal that do change its total.
        Empty means the total is strictly conserved and can replace one state variable, as the
        lectures do with total ligand or total enzyme in a closed system. Nonempty means the
        total obeys a balance whose right-hand side is exactly this list.
    """
    coefficients: dict
    text: str
    changed_by: list
    total_text: str

    @property
    def is_conserved(self):
        return not self.changed_by

    @property
    def balance_text(self):
        if self.is_conserved:
            return f'{self.total_text}: conserved'
        return f'{self.total_text}: changes only through {", ".join(self.changed_by)}'


def smooth_step(x, threshold, width):
    """A smooth switch from 0 below ``threshold`` to 1 above it, over a scale ``width``.

    Use it in a rate law in place of ``1 if x > threshold else 0``. Integrators assume the
    right-hand side is smooth; a hard switch stalls the step-size control, can make the solver
    chatter when the state hovers near the threshold, and can hide a mistimed crossing. A steep
    logistic keeps the solver honest and is also the more plausible biology, since no process
    responds discontinuously to a concentration. Works on scalars and arrays.
    """
    return 1.0 / (1.0 + np.exp(-(np.asarray(x, dtype=float) - threshold) / width))


def _latex_symbol(name):
    parts = name.split('_')
    head = parts[0]
    head_latex = head if len(head) == 1 else f'\\mathrm{{{head}}}'
    if len(parts) == 1:
        return head_latex
    return f"{head_latex}_{{\\mathrm{{{','.join(parts[1:])}}}}}"


def _latex_species(name):
    return f'[{_latex_symbol(name)}]'


def _mass_action_text(rate_constant, stoichiometry):
    parts = [rate_constant]
    parts_latex = [_latex_symbol(rate_constant)]
    for species, coefficient in stoichiometry.items():
        parts.append(f'[{species}]' if coefficient == 1 else f'[{species}]^{coefficient}')
        parts_latex.append(_latex_species(species) if coefficient == 1
                           else f'{_latex_species(species)}^{{{coefficient}}}')
    return ' × '.join(parts), ' \\, '.join(parts_latex)


def _integer_left_null_space(matrix):
    """Integer basis of {c : c^T S = 0} by exact row reduction of S^T over the rationals."""
    rows, columns = matrix.shape
    system = [[Fraction(int(matrix[i, j])) for i in range(rows)] for j in range(columns)]  # S^T
    pivot_columns = []
    pivot_row = 0
    for column in range(rows):
        pivot = next((r for r in range(pivot_row, len(system)) if system[r][column] != 0), None)
        if pivot is None:
            continue
        system[pivot_row], system[pivot] = system[pivot], system[pivot_row]
        scale = system[pivot_row][column]
        system[pivot_row] = [value / scale for value in system[pivot_row]]
        for r in range(len(system)):
            if r != pivot_row and system[r][column] != 0:
                factor = system[r][column]
                system[r] = [a - factor * b for a, b in zip(system[r], system[pivot_row])]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(system):
            break
    free_columns = [c for c in range(rows) if c not in pivot_columns]
    basis = []
    for free in free_columns:
        vector = [Fraction(0)] * rows
        vector[free] = Fraction(1)
        for r, pivot_column in enumerate(pivot_columns):
            vector[pivot_column] = -system[r][free]
        denominators = [value.denominator for value in vector]
        common = 1
        for d in denominators:
            common = common * d // _gcd(common, d)
        basis.append([int(value * common) for value in vector])
    return _nonnegative_basis(basis)


def _nonnegative_basis(basis):
    """Combine basis vectors so that each has nonnegative entries where a combination allows it.

    Row reduction returns a valid basis, but a vector such as R_s - A_b is less useful than
    R_s + C_s when both span the same space. Each vector is replaced by the sparsest nonnegative
    combination of itself with small multiples of the others, when one exists.
    """
    vectors = [list(v) for v in basis]
    for i, vector in enumerate(vectors):
        if all(value <= 0 for value in vector):
            vector = [-value for value in vector]
            vectors[i] = vector
        if all(value >= 0 for value in vector):
            continue
        best = None
        for j, other in enumerate(vectors):
            if j == i:
                continue
            for multiple in (1, -1, 2, -2, 3, -3):
                candidate = [a + multiple * b for a, b in zip(vector, other)]
                if any(value < 0 for value in candidate) or not any(candidate):
                    continue
                support = sum(1 for value in candidate if value)
                if best is None or support < best[0]:
                    best = (support, candidate)
        if best is not None:
            vectors[i] = best[1]
    normalized = []
    for vector in vectors:
        divisor = 0
        for value in vector:
            divisor = _gcd(divisor, abs(value))
        normalized.append([value // divisor for value in vector] if divisor else vector)
    return normalized


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class CompiledModel:
    """A validated network with its parameters: derivatives, stoichiometry, equations, checks."""

    def __init__(self, network, parameters, time_units):
        if not isinstance(parameters, Parameters):
            raise ValueError("compile() needs a Parameters table.")
        if time_units not in TIME_UNITS_IN_SECONDS:
            raise ValueError(f"time_units must be one of {list(TIME_UNITS_IN_SECONDS)}, not '{time_units}'.")
        self.network = network
        self.parameters = parameters
        self.time_units = time_units
        self.species_names = list(network.species)
        self.accumulator_names = list(network.accumulators)
        self.state_names = self.species_names + self.accumulator_names
        self._index = {name: i for i, name in enumerate(self.state_names)}
        self.reactions = list(network.reactions)
        self.references = [network.reference_of(reaction) for reaction in self.reactions]
        self._validate()
        self.stoichiometry = self._build_stoichiometry()
        self._cell_losses = [
            (j, species) for j, reaction in enumerate(self.reactions)
            for species, net in reaction.net_stoichiometry.items()
            if net < 0 and isinstance(network.species[species], PopulationCount)
        ]
        self._per_cell_species_of = {
            compartment.name: [name for name in self.species_names
                               if isinstance(network.species[name], ChemicalSpecies)
                               and network.species[name].compartment is compartment]
            for compartment in network.replicated_compartments
        }

    # ----- validation ---------------------------------------------------------------------

    def _validate(self):
        network, k, time_units = self.network, self.parameters, self.time_units
        for compartment in network.compartments.values():
            if compartment.is_replicated:
                count = compartment.count_variable
                if (count not in network.species
                        or not isinstance(network.species[count], PopulationCount)
                        or network.species[count].population is not compartment):
                    raise ValueError(
                        f"Compartment '{compartment.name}' must own the population count '{count}'. "
                        "Declare it through add_compartment."
                    )
        for reaction, (kind, compartment) in zip(self.reactions, self.references):
            for name in reaction.parameter_names():
                if name not in k:
                    raise ValueError(f"Reaction '{reaction.name}' uses parameter '{name}', which is not declared.")
            reference_units = self._reference_units(kind) / Unit(time_units)
            if reaction.forward_rate_constant is not None:
                self._check_mass_action_units(reaction, reaction.forward_rate_constant, reaction.reactants,
                                              reference_units, 'forward')
            if reaction.rate_function is not None or reaction.reverse_rate_function is not None:
                check_time_units(reaction.rate_units, time_units, f"Reaction '{reaction.name}' rate_units")
                if reaction.rate_units != reference_units:
                    raise ValueError(
                        f"Reaction '{reaction.name}': custom rate_units '{reaction.rate_units}' must be "
                        f"'{reference_units}' in this reaction's reference compartment."
                    )
            if reaction.reverse_rate_constant is not None:
                self._check_mass_action_units(reaction, reaction.reverse_rate_constant, reaction.products,
                                              reference_units, 'reverse')
        for term in network.direct_terms:
            check_time_units(term.units, time_units, f"Direct term on '{term.species}'")
            expected = unit_of_species(network.species[term.species].units) / Unit(time_units)
            if term.units != expected:
                raise ValueError(
                    f"Direct term on '{term.species}' has units '{term.units}' but must be '{expected}' "
                    f"(the species' declared units per {time_units})."
                )
        for accumulator in network.accumulators.values():
            check_time_units(accumulator.units, time_units, f"Accumulator '{accumulator.name}'")

    @staticmethod
    def _reference_units(kind):
        return {'cells': Unit('cells'), 'per_cell': Unit('#/cell'), 'bulk': Unit('M')}[kind]

    def _check_mass_action_units(self, reaction, constant_name, stoichiometry, reference_units, direction):
        units = self.parameters.units(constant_name)
        check_time_units(units, self.time_units, f"Parameter '{constant_name}'")
        for species, coefficient in stoichiometry.items():
            units = units * unit_of_species(self.network.species[species].units) ** coefficient
        if units != reference_units:
            factors = ' × '.join([constant_name] + [f'[{s}]' if c == 1 else f'[{s}]^{c}' for s, c in stoichiometry.items()])
            raise ValueError(
                f"Reaction '{reaction.name}' ({direction}): {factors} has units '{units}', but a rate in this "
                f"reaction's reference compartment must be '{reference_units}'. Check the units of "
                f"'{constant_name}' ({self.parameters.units(constant_name)}); convert_rate_constant may help."
            )

    # ----- structure ----------------------------------------------------------------------

    def _build_stoichiometry(self):
        matrix = np.zeros((len(self.species_names), len(self.reactions)), dtype=int)
        for j, reaction in enumerate(self.reactions):
            for species, net in reaction.net_stoichiometry.items():
                matrix[self._index[species], j] = net
        return matrix

    def stoichiometry_table(self):
        """The stoichiometry matrix as a pandas DataFrame, species by reaction."""
        import pandas as pd
        return pd.DataFrame(self.stoichiometry, index=self.species_names,
                            columns=[reaction.name for reaction in self.reactions])

    def species(self, name):
        if name not in self.network.species:
            raise KeyError(f"'{name}' is not a species of this model.")
        return self.network.species[name]

    def compartment(self, name):
        if name not in self.network.compartments:
            raise KeyError(f"'{name}' is not a compartment of this model.")
        return self.network.compartments[name]

    def _factor_to_molecules(self, kind, compartment, values):
        if kind == 'cells':
            return 1.0
        if kind == 'per_cell':
            return values[compartment.count_variable]
        return compartment.volume * AVOGADRO

    def _species_factor(self, name, values):
        species = self.network.species[name]
        if isinstance(species, PopulationCount):
            return 1.0
        if species.compartment.is_replicated:
            return values[species.compartment.count_variable]
        return species.compartment.volume * AVOGADRO

    # ----- conversions --------------------------------------------------------------------

    def declared_from_totals(self, totals):
        """Dictionary of species and accumulators in declared units, from a totals vector or matrix."""
        totals = np.asarray(totals)
        network = self.network
        values = {}
        for name in self.species_names:
            if isinstance(network.species[name], PopulationCount):
                values[name] = totals[self._index[name]]
        for name in self.species_names:
            species = network.species[name]
            if isinstance(species, ChemicalSpecies):
                values[name] = totals[self._index[name]] / self._species_factor(name, values)
        for name in self.accumulator_names:
            values[name] = totals[self._index[name]]
        return values

    def state(self, values):
        """Wrap a state dictionary or vector in the species' declared units."""
        if not isinstance(values, Mapping):
            values = dict(zip(self.state_names, np.asarray(values)))
        return State(values, self)

    def totals_from_declared(self, declared):
        """Vector of totals (molecules, cells, accumulator units) from declared-unit values."""
        totals = np.array(np.broadcast_arrays(*[
            np.asarray(declared.get(name, 0.0), dtype=float) for name in self.state_names
        ]))
        for name in self.state_names:
            if name in self.network.accumulators:
                continue
            species = self.network.species[name]
            totals[self._index[name]] *= self._species_factor(name, declared)
        return totals

    # ----- Model interface -----------------------------------------------------------------

    def validate_initial_values(self, initial_values):
        for name in initial_values:
            if name not in self._index:
                raise ValueError(f"initial_values names '{name}', which is not a declared species or accumulator.")
        for compartment in self.network.replicated_compartments:
            count = compartment.count_variable
            value = initial_values.get(count, 0.0)
            if self._per_cell_species_of[compartment.name] and value <= 0:
                raise ValueError(
                    f"initial_values must give a positive cell count for '{count}' (compartment '{compartment.name}')."
                )
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"initial_values must give a finite nonnegative count for '{count}'.")

    def initial_state(self, initial_values):
        self.validate_initial_values(initial_values)
        return np.array([initial_values.get(name, 0.0) for name in self.state_names], dtype=float)

    def validate_impulse(self, impulse):
        if impulse.species not in self.network.species:
            if impulse.species in self.network.accumulators:
                raise ValueError(f"Impulses can only be added to species, not to accumulator '{impulse.species}'.")
            raise ValueError(f"Impulse at {impulse.time} targets '{impulse.species}', which is not a state variable.")
        species = self.species(impulse.species)
        if isinstance(species, PopulationCount):
            allowed = ('cells',)
        elif species.compartment.is_replicated:
            allowed = ('#/cell', 'molecules')
        else:
            allowed = ('mol', 'molecules', 'M')
        if impulse.units not in allowed:
            raise ValueError(f"Impulse on '{impulse.species}' ({species.units}) has units '{impulse.units}'; "
                             f"use one of {allowed}.")

    def impulse_increment(self, impulse, state):
        species = self.network.species[impulse.species]
        if impulse.units == species.units:
            return impulse.amount
        values = dict(zip(self.state_names, state))
        factor = self._species_factor(impulse.species, values)
        if impulse.units == 'molecules':
            return impulse.amount / factor
        if impulse.units == 'mol':
            return impulse.amount * AVOGADRO / factor
        raise ValueError(f"Unhandled impulse units '{impulse.units}'.")

    def apply_impulse(self, impulse, state):
        """Apply an impulse; changing a count alone retains its molecular totals."""
        updated = state.copy()
        index = self._index[impulse.species]
        updated[index] += self.impulse_increment(impulse, state)
        for compartment in self.network.replicated_compartments:
            if compartment.count_variable == impulse.species:
                names = self._per_cell_species_of[compartment.name]
                if names and updated[index] <= 0:
                    raise ValueError(f"Impulse on '{impulse.species}' must leave a positive cell count.")
                if updated[index] < 0:
                    raise ValueError(f"Impulse on '{impulse.species}' must leave a nonnegative count.")
                for name in names:
                    updated[self._index[name]] *= state[index] / updated[index]
        return updated

    def reaction_rates(self, values):
        """Forward-minus-reverse rate of every reaction, in molecules (or cells) per time."""
        s = State(values, self)
        k = self.parameters
        rates = np.zeros(len(self.reactions))
        for j, (reaction, (kind, compartment)) in enumerate(zip(self.reactions, self.references)):
            factor = self._factor_to_molecules(kind, compartment, values)
            rate = reaction.forward_rate(s, k)
            if reaction.is_reversible:
                rate = rate - reaction.reverse_rate(s, k)
            rates[j] = rate * factor
        return rates

    def rhs(self, t, state):
        """Derivatives of the state vector, in declared units per simulation time."""
        values = dict(zip(self.state_names, state))
        rates = self.reaction_rates(values)
        derivative = np.zeros(len(state))
        derivative[:len(self.species_names)] = self.stoichiometry @ rates
        s = State(values, self)
        k = self.parameters
        direct_rates = [(term.species, term.function(s, k)) for term in self.network.direct_terms]
        count_rates = {name: derivative[self._index[name]] for name in self.species_names
                       if isinstance(self.species(name), PopulationCount)}
        for name, rate in direct_rates:
            if name in count_rates:
                count_rates[name] += rate
        for j, count_species in self._cell_losses:
            count_rates[count_species] -= self.stoichiometry[self._index[count_species], j] * rates[j]
        for name in self.species_names:
            species = self.species(name)
            index = self._index[name]
            derivative[index] /= self._species_factor(name, values)
            if isinstance(species, ChemicalSpecies) and species.compartment.is_replicated:
                count = species.compartment.count_variable
                derivative[index] -= values[name] * count_rates[count] / values[count]
        for name, rate in direct_rates:
            derivative[self._index[name]] += rate
        for name, accumulator in self.network.accumulators.items():
            derivative[self._index[name]] += accumulator.rate_function(s, k)
        return derivative

    def atol_vector(self, initial_state, atol, accumulator_atol):
        defaults = {'M': 1e-20, '#/cell': 1e-8, 'cells': 1e-8}
        tolerances = {name: defaults[self.species(name).units] for name in self.species_names}
        for name in self.accumulator_names:
            tolerances[name] = accumulator_atol
        if isinstance(atol, Mapping):
            unknown = set(atol) - set(self.state_names)
            if unknown:
                raise ValueError(f"atol names unknown state variables: {sorted(unknown)}")
            tolerances.update(atol)
        elif atol is not None:
            for name in self.species_names:
                tolerances[name] = atol
        vector = np.array([tolerances[name] for name in self.state_names], dtype=float)
        if not np.all(np.isfinite(vector)) or np.any(vector <= 0):
            raise ValueError('Absolute tolerances must be finite and positive.')
        return vector, dict(zip(self.state_names, vector))

    def package(self, t, state_matrix, condition_name, atol_used, impulse_times):
        network = self.network
        declared = dict(zip(self.state_names, state_matrix.T))
        values = {name: np.asarray(declared[name]) for name in self.species_names}
        accumulators = {name: np.asarray(declared[name]) for name in self.accumulator_names}
        totals = dict(zip(self.state_names, self.totals_from_declared(declared)))
        state = State(declared, self)
        observables = {}
        for name, observable in network.observables.items():
            observables[name] = np.asarray(observable.function(state, self.parameters), dtype=float)
            if observables[name].ndim == 0:
                observables[name] = np.full(t.shape, observables[name].item())
            if observables[name].shape != t.shape:
                raise ValueError(
                    f"Observable '{name}' returned shape {observables[name].shape}; return a scalar constant or "
                    f"one value per time point ({t.shape}). Write varying quantities with NumPy array operations."
                )
        units = {**{n: sp.units for n, sp in network.species.items()},
                 **{n: o.units for n, o in network.observables.items()},
                 **{n: str(a.units) for n, a in network.accumulators.items()}}
        descriptions = {**{n: sp.description for n, sp in network.species.items()},
                        **{n: o.description for n, o in network.observables.items()},
                        **{n: a.description for n, a in network.accumulators.items()}}
        return SimulationResult(condition_name, t, self.time_units, values, units, descriptions, totals,
                                observables, accumulators, atol_used, impulse_times, model=self)

    # ----- derivatives for checks ---------------------------------------------------------

    def derivative_function(self):
        """Callable (t, state_dict) -> dict of d/dt in declared units, for checks or solve_ivp."""
        def derivative(t, state):
            values = self.initial_state(state)
            return dict(zip(self.state_names, self.rhs(t, values)))
        return derivative

    # ----- conservation -------------------------------------------------------------------

    def moieties(self):
        """The combinations of species that the reactions shuffle but neither create nor destroy.

        Found from the stoichiometry matrix: a vector c with c·S' = 0, where S' keeps the
        reactions with reactants and products on both sides, is unchanged by every one of them.
        Each Moiety lists what does change its total: the source and sink reactions (∅ on one
        side), direct rate terms, and removal of contents with dead cells.

        In a closed system, a binding reaction in a tube or an enzyme assay, the list is empty:
        the total is strictly conserved and can replace one state variable, as the lectures do
        with total ligand or total enzyme. In an open system, a cell that makes and degrades its
        receptors and a blood compartment that clears drug, the list is the right-hand side of
        the balance equation for that total, which is what the lecture writes when it adds the
        surface receptor and complex equations into one balance driven by synthesis and
        endocytosis.

        The computation is stoichiometric: it cannot see a quantity that is conserved only
        because rate laws cancel, and a nonnegative vector it returns is a moiety in the chemical
        sense only when the species really are forms of one molecular part, which is the case for
        receptor, ligand and payload totals in this project.
        """
        kept = [j for j, reaction in enumerate(self.reactions) if not reaction.is_source_or_sink]
        matrix = self.stoichiometry[:, kept] if kept else np.zeros((len(self.species_names), 0), dtype=int)
        moieties = []
        for coefficients in _integer_left_null_space(matrix):
            weights = dict(zip(self.species_names, coefficients))
            terms = []
            for name, c in zip(self.species_names, coefficients):
                if c == 0:
                    continue
                terms.append(f'{"" if c == 1 else ("-" if c == -1 else f"{c} ")}{name}')
            text = ' + '.join(terms).replace('+ -', '- ')
            total_terms = []
            for name, coefficient in zip(self.species_names, coefficients):
                if coefficient == 0:
                    continue
                species = self.network.species[name]
                if isinstance(species, PopulationCount):
                    total = name
                elif species.compartment.is_replicated:
                    total = f'[{name}] {species.compartment.count_variable}'
                else:
                    total = f'[{name}] N_Av V_{species.compartment.name}'
                prefix = '' if coefficient == 1 else ('-' if coefficient == -1 else f'{coefficient} ')
                total_terms.append(prefix + total)
            total_text = ' + '.join(total_terms).replace('+ -', '- ')
            changed_by = []
            c = np.array(coefficients)
            for j, reaction in enumerate(self.reactions):
                if reaction.is_source_or_sink and c @ self.stoichiometry[:, j] != 0:
                    changed_by.append(reaction.name)
            for j, count_species in self._cell_losses:
                if any(weights[name] for compartment in self.network.replicated_compartments
                       if compartment.count_variable == count_species
                       for name in self._per_cell_species_of[compartment.name]):
                    changed_by.append(f'{self.reactions[j].name} (removes contents)')
            for term in self.network.direct_terms:
                if weights[term.species]:
                    changed_by.append(f'direct term on {term.species}')
            moieties.append(Moiety(weights, text, changed_by, total_text))
        return moieties

    # ----- display ------------------------------------------------------------------------

    def _forward_text(self, reaction):
        if reaction.rate_function is not None:
            latex = reaction.rate_expression_latex or f'\\text{{{reaction.rate_expression}}}'
            return reaction.rate_expression, latex
        return _mass_action_text(reaction.forward_rate_constant, reaction.reactants)

    def _reverse_text(self, reaction):
        if reaction.reverse_rate_function is not None:
            latex = reaction.reverse_rate_expression_latex or f'\\text{{{reaction.reverse_rate_expression}}}'
            return reaction.reverse_rate_expression, latex
        return _mass_action_text(reaction.reverse_rate_constant, reaction.products)

    def _lhs(self, name):
        species = self.network.species[name]
        if (self.network.replicated_compartments and isinstance(species, ChemicalSpecies)
                and not species.compartment.is_replicated):
            volume = f'V_{species.compartment.name}'
            return (f'N_Av {volume} d[{name}]/dt',
                    f'N_{{\\mathrm{{Av}}}} {_latex_symbol(volume)} \\frac{{d{_latex_species(name)}}}{{dt}}')
        return f'd[{name}]/dt', f'\\frac{{d{_latex_species(name)}}}{{dt}}'

    def _term_factor(self, name, kind, compartment):
        species = self.network.species[name]
        if isinstance(species, PopulationCount) or species.compartment.is_replicated:
            return '', ''
        if not self.network.replicated_compartments:
            if compartment == species.compartment:
                return '', ''
            reference_volume = f'V_{compartment.name}'
            species_volume = f'V_{species.compartment.name}'
            return (f'{reference_volume}/{species_volume} × ',
                    f'\\frac{{{_latex_symbol(reference_volume)}}}{{{_latex_symbol(species_volume)}}} \\, ')
        if kind == 'per_cell':
            count = compartment.count_variable
            return f'{count} × ', f'{_latex_symbol(count)} \\, '
        volume = f'V_{compartment.name}'
        return f'N_Av {volume} × ', f'N_{{\\mathrm{{Av}}}} {_latex_symbol(volume)} \\, '

    def rate_equations(self):
        """Per species, the printed rate equation as structured terms (text and LaTeX)."""
        terms = {name: [] for name in self.species_names}
        counts_with_dilution = set()
        for reaction, (kind, compartment) in zip(self.reactions, self.references):
            forward_text, forward_latex = self._forward_text(reaction)
            for name, net in reaction.net_stoichiometry.items():
                prefix_text, prefix_latex = self._term_factor(name, kind, compartment)
                magnitude = '' if abs(net) == 1 else f'{abs(net)} × '
                magnitude_latex = '' if abs(net) == 1 else f'{abs(net)} \\, '
                sign = '+' if net > 0 else '-'
                terms[name].append(RateTerm(f'{sign} {magnitude}{prefix_text}{forward_text}',
                                            f'{sign} {magnitude_latex}{prefix_latex}{forward_latex}', 'reaction'))
                if reaction.is_reversible:
                    reverse_text, reverse_latex = self._reverse_text(reaction)
                    reverse_sign = '-' if net > 0 else '+'
                    terms[name].append(RateTerm(f'{reverse_sign} {magnitude}{prefix_text}{reverse_text}',
                                                f'{reverse_sign} {magnitude_latex}{prefix_latex}{reverse_latex}', 'reverse'))
                if kind == 'cells' and net > 0:
                    counts_with_dilution.add(name)
        counts_with_dilution.update(term.species for term in self.network.direct_terms
                                   if isinstance(self.network.species[term.species], PopulationCount))
        for compartment in self.network.replicated_compartments:
            count = compartment.count_variable
            if count in counts_with_dilution:
                for name in self._per_cell_species_of[compartment.name]:
                    terms[name].append(RateTerm(
                        f'- (1/{count})(d{count}/dt)* × [{name}]',
                        f'- \\frac{{1}}{{{_latex_symbol(count)}}}\\left(\\frac{{d{_latex_symbol(count)}}}{{dt}}\\right)'
                        f'^{{*}} {_latex_species(name)}', 'dilution'))
        for term in self.network.direct_terms:
            terms[term.species].append(RateTerm(f'+ {term.expression} [direct term]',
                                                f'+ \\text{{{term.expression}}} \\;\\text{{[direct term]}}', 'direct'))
        equations = {}
        for name in self.species_names:
            lhs_text, lhs_latex = self._lhs(name)
            equations[name] = RateEquation(name, lhs_text, lhs_latex, terms[name])
        return equations

    @staticmethod
    def _join_terms(parts):
        if not parts:
            return '0'
        joined = ' '.join(parts)
        return joined[2:] if joined.startswith('+ ') else joined

    def rate_equation_text(self):
        return '\n'.join(f"{eq.lhs_text} = {self._join_terms([term.text for term in eq.terms])}"
                         for eq in self.rate_equations().values())

    def print_rate_equations(self):
        """Print the rate equations; in a notebook, also render them with LaTeX."""
        equations = self.rate_equations()
        try:
            from IPython.display import display, Math
            in_notebook = True
        except ImportError:
            in_notebook = False
        if self.network.replicated_compartments:
            counts = ', '.join(c.count_variable for c in self.network.replicated_compartments)
            print(f"Per-cell species are in molecules per cell; {counts} is the cell count (= ρV in the lecture). "
                  "Molar balances are written multiplied through by N_Av V, as in the lecture.")
        for name, equation in equations.items():
            text = f"{equation.lhs_text} = {self._join_terms([term.text for term in equation.terms])}"
            description = self.network.species[name].description
            if in_notebook:
                display(Math(f"{equation.lhs_latex} = {self._join_terms([term.latex for term in equation.terms])}"))
                print(f"    {text}    ({description})")
            else:
                print(f"{text}    ({description})")
        if any(term.kind == 'dilution' for equation in equations.values() for term in equation.terms):
            print("* The cell-count derivative includes growth reactions and direct count terms; "
                  "cell-loss reactions leave per-cell amounts unchanged.")
        if self.network.direct_terms:
            names = sorted({term.species for term in self.network.direct_terms})
            print(f"Direct terms were added to {names}; moiety balances include them as sources or sinks.")

    def print_parameters(self):
        print(f"Parameters (time in {self.time_units}):")
        print(self.parameters)


class HandWrittenModel:
    """A problem-set-style ``dydt(t, state_dict) -> dict`` as a Model, with exact impulses and plotting.

    No unit bookkeeping or validation is applied; impulses must use units 'raw'; observables and
    accumulators are not available.
    """

    def __init__(self, derivative_function, species, time_units='min'):
        if not species:
            raise ValueError("HandWrittenModel needs species=[...] listing the state names in order.")
        self.derivative_function = derivative_function
        self.state_names = list(species)
        self.time_units = time_units

    def validate_initial_values(self, initial_values):
        for name in initial_values:
            if name not in self.state_names:
                raise ValueError(f"initial_values names '{name}', which is not in species.")

    def initial_state(self, initial_values):
        self.validate_initial_values(initial_values)
        return np.array([initial_values.get(name, 0.0) for name in self.state_names], dtype=float)

    def validate_impulse(self, impulse):
        if impulse.species not in self.state_names:
            raise ValueError(f"Impulse at {impulse.time} targets '{impulse.species}', which is not a state variable.")
        if impulse.units != 'raw':
            raise ValueError("With a hand-written derivative function, impulses must use units 'raw'.")

    def impulse_increment(self, impulse, state):
        return impulse.amount

    def apply_impulse(self, impulse, state):
        updated = state.copy()
        updated[self.state_names.index(impulse.species)] += self.impulse_increment(impulse, state)
        return updated

    def rhs(self, t, y):
        derivatives = self.derivative_function(t, dict(zip(self.state_names, y)))
        return [derivatives[name] for name in self.state_names]

    def atol_vector(self, initial_state, atol, accumulator_atol):
        if atol is None:
            nonzero = [abs(v) for v in initial_state if v != 0]
            atol = 1e-9 * min(nonzero) if nonzero else 1e-9
        return np.full(len(self.state_names), float(atol)), atol

    def package(self, t, state_matrix, condition_name, atol_used, impulse_times):
        values = {name: state_matrix[:, i] for i, name in enumerate(self.state_names)}
        return SimulationResult(condition_name, t, self.time_units, values, {n: 'raw' for n in self.state_names},
                                {n: n for n in self.state_names}, dict(values), {}, {}, atol_used, impulse_times,
                                model=self)


# ===== Impulses ==================================================================================

Impulse = namedtuple('Impulse', ['time', 'species', 'amount', 'units'])
Impulse.__doc__ = """An instantaneous addition to one species at one time (time in the simulation's units).

units: 'mol', 'molecules' or 'M' for a molar species; '#/cell' or 'molecules' for a per-cell
species; 'cells' for the cell count; 'raw' with a HandWrittenModel.
"""


@dataclass
class Regimen:
    """Repeated impulses: ``amount`` of ``species`` every ``interval``, starting at ``start``.

    Impulses are given at start, start + interval, ... while before ``end`` (the simulation's last
    time when None), and at most ``number_of_impulses`` of them when given.
    A regimen with no impulses in this period returns an empty list.
    """
    species: str
    amount: float
    units: str
    interval: float
    start: float = 0.0
    end: float = None
    number_of_impulses: int = None

    def impulses(self, last_time):
        if self.interval <= 0:
            raise ValueError("Regimen interval must be positive.")
        end = last_time if self.end is None else self.end
        times = []
        time = self.start
        while time < end and (self.number_of_impulses is None or len(times) < self.number_of_impulses):
            times.append(time)
            time += self.interval
        return [Impulse(time, self.species, self.amount, self.units) for time in times]


# ===== Simulation ==============================================================================

class SimulationResult:
    """Time series in declared units, with totals, observables, and accumulators.

    ``result['A_b']`` returns a species, observable, or accumulator array; ``result['t']`` the
    time axis. ``result.totals['A_b']`` is the total in molecules. ``result.time_in('day')``
    converts the axis. ``result.impulse_times`` lists when impulses were applied.
    """

    def __init__(self, condition_name, t, time_units, values, units, descriptions, totals,
                 observables, accumulators, atol_used, impulse_times, model=None):
        self.condition_name = condition_name
        self.t = t
        self.time_units = time_units
        self.values = values
        self.units = units
        self.descriptions = descriptions
        self.totals = totals
        self.observables = observables
        self.accumulators = accumulators
        self.atol_used = atol_used
        self.impulse_times = impulse_times
        self.model = model

    def __getitem__(self, name):
        if name == 't':
            return self.t
        for table in (self.values, self.observables, self.accumulators):
            if name in table:
                return table[name]
        raise KeyError(f"'{name}' is not a species, observable, accumulator, or 't' in these results.")

    def __contains__(self, name):
        return name == 't' or any(name in table for table in (self.values, self.observables, self.accumulators))

    def names(self):
        return list(self.values) + list(self.observables) + list(self.accumulators)

    def time_in(self, units):
        if units not in TIME_UNITS_IN_SECONDS:
            raise ValueError(f"units must be one of {list(TIME_UNITS_IN_SECONDS)}.")
        return self.t * TIME_UNITS_IN_SECONDS[self.time_units] / TIME_UNITS_IN_SECONDS[units]

    def in_days(self):
        return self.time_in('day')

    def in_hours(self):
        return self.time_in('hr')

    def at(self, name, time):
        """Value of a series at the last grid point at or before ``time``."""
        index = np.flatnonzero(self.t <= time)
        if len(index) == 0:
            raise ValueError(f"time {time} precedes the first grid point {self.t[0]}.")
        return self[name][index[-1]]


class Simulation:
    """Integrate a model, with exact impulses, on an output grid.

    model : a CompiledModel from ``kinetic_model.compile(...)`` or a HandWrittenModel.
    t : strictly increasing output grid in the model's time units. Impulse times not on the grid
        are inserted. Impulses after the initial time have before-and-after states; an impulse
        at the initial time is applied before the first recorded state.
    initial_values : declared units; unlisted states start at zero. Counts for compartments
        with per-cell molecular species must be positive; count-only populations may start at zero.
    impulses : Impulse and Regimen objects, mixed freely.
    method, rtol, atol : solver controls. For reaction networks, ``atol`` is in declared units:
        a scalar for all species, or a dictionary of per-state overrides. Defaults are 1e-20 M,
        1e-8 molecules/cell, and 1e-8 cells. ``result.atol_used`` reports each state's tolerance.
    accumulator_atol : absolute tolerance for accumulators in their own units.
    """

    def __init__(self, condition_name, t, model, initial_values=None, impulses=(), method='LSODA',
                 rtol=1e-8, atol=None, accumulator_atol=1e-30):
        if not hasattr(model, 'rhs') or not hasattr(model, 'state_names'):
            raise ValueError("Simulation needs a model: kinetic_model.compile(parameters) or a HandWrittenModel.")
        self.condition_name = condition_name
        self.t = np.asarray(t, dtype=float)
        self.model = model
        self.initial_values = dict(initial_values or {})
        self.impulses = list(impulses)
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.accumulator_atol = accumulator_atol

    @property
    def time_units(self):
        return self.model.time_units

    def expanded_impulses(self):
        impulses = []
        for item in self.impulses:
            if isinstance(item, Regimen):
                impulses.extend(item.impulses(self.t[-1]))
            elif isinstance(item, Impulse):
                impulses.append(item)
            else:
                raise ValueError(f"impulses must contain Impulse or Regimen objects, not {type(item).__name__}.")
        return impulses

    def validate(self):
        if self.t.ndim != 1 or len(self.t) < 2 or np.any(np.diff(self.t) <= 0):
            raise ValueError("t must be a strictly increasing array with at least two points.")
        self.model.validate_initial_values(self.initial_values)
        for impulse in self.expanded_impulses():
            if impulse.units not in IMPULSE_UNITS:
                raise ValueError(f"Impulse units '{impulse.units}' not recognized; use one of {IMPULSE_UNITS}.")
            self.model.validate_impulse(impulse)
            if not (self.t[0] <= impulse.time <= self.t[-1]):
                raise ValueError(f"Impulse at {impulse.time} lies outside the grid [{self.t[0]}, {self.t[-1]}].")

    def simulate(self):
        self.validate()
        model = self.model
        state = model.initial_state(self.initial_values)
        atol_vector, atol_used = model.atol_vector(state, self.atol, self.accumulator_atol)
        impulses_by_time = {}
        for impulse in sorted(self.expanded_impulses(), key=lambda d: d.time):
            impulses_by_time.setdefault(impulse.time, []).append(impulse)
        boundaries = sorted(set([self.t[0], self.t[-1]] + list(impulses_by_time)))

        def apply_impulses(at_time, state):
            for impulse in impulses_by_time.get(at_time, []):
                state = model.apply_impulse(impulse, state)
            return state

        if self.t[0] in impulses_by_time:
            state = apply_impulses(self.t[0], state)
        times, states = [], []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            grid = self.t[(self.t >= start) & (self.t <= end)]
            if len(grid) == 0 or grid[0] != start:
                grid = np.concatenate([[start], grid])
            if grid[-1] != end:
                grid = np.concatenate([grid, [end]])
            solution = solve_ivp(model.rhs, (start, end), state, t_eval=grid, method=self.method,
                                 rtol=self.rtol, atol=atol_vector)
            if not solution.success:
                raise ValueError(
                    f"Integration failed for '{self.condition_name}' between t = {start:g} and {end:g} "
                    f"{self.time_units}: {solution.message}"
                )
            if len(solution.t) != len(grid):
                raise ValueError(
                    f"Integration for '{self.condition_name}' returned {len(solution.t)} of {len(grid)} grid "
                    f"points between t = {start:g} and {end:g}."
                )
            times.extend(solution.t)
            states.extend(solution.y.T)
            state = solution.y[:, -1].copy()
            if end in impulses_by_time and end != self.t[0]:
                state = apply_impulses(end, state)
                if end == self.t[-1]:
                    times.append(end)
                    states.append(state.copy())
        result = model.package(np.array(times), np.array(states), self.condition_name, atol_used, sorted(impulses_by_time))
        return result

    def print_rate_equations(self):
        if isinstance(self.model, CompiledModel):
            self.model.print_rate_equations()
        else:
            print("This simulation uses a hand-written derivative function; no generated equations.")

    def print_initial_conditions(self):
        print(f"Initial conditions for '{self.condition_name}':")
        for name in self.model.state_names:
            print(f"  {name:<12} {self.initial_values.get(name, 0.0):>12.4g}")
        for impulse in sorted(self.expanded_impulses(), key=lambda d: d.time):
            print(f"  impulse at t = {impulse.time:g} {self.time_units}: {impulse.amount:.4g} {impulse.units} of {impulse.species}")


# ===== Plotting ================================================================================

def make_plot_look_good(axes):
    """Apply consistent styling to plot axes"""
    axes.axhline(0, color=[0.2, 0.2, 0.2], linewidth=1)
    axes.axvline(0, color=[0.2, 0.2, 0.2], linewidth=1)
    axes.grid(True, which='both', linestyle='--', linewidth=0.5)
    for spine in axes.spines.values():
        spine.set_visible(False)


def set_axes_labels(axes, x_label, y_label, title, legend_title=None):
    """Set axes labels and title with consistent styling"""
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    if legend_title is not None:
        axes.legend(title=legend_title, bbox_to_anchor=(1, 1))
    make_plot_look_good(axes)


def plot_simulations(results, variables=None, time_units=None, figsize=(15, 10), log_scale=False,
                     overall_title=None, legend_style='shared', time_range=None, columns=3,
                     unit_conversions=None):
    """Plot one or more SimulationResult objects, one panel per variable.

    ``variables`` may name species, observables, or accumulators; the default is every species.
    ``log_scale`` is False, True or 'loglog', 'logx', or 'logy'. ``time_range`` is (min, max) in
    ``time_units``. ``unit_conversions`` maps a plotted variable to ``(factor, units_label)``: its
    values are multiplied by ``factor`` and its axis is labeled ``units_label``.
    """
    results_list = [results] if isinstance(results, SimulationResult) else list(results)
    if not results_list:
        raise ValueError("results cannot be empty")
    first = results_list[0]
    if variables is None:
        variables = list(first.values)
    for name in variables:
        if name not in first:
            raise ValueError(f"'{name}' is not in the results; available: {first.names()}")
    unit_conversions = unit_conversions or {}
    for name, conversion in unit_conversions.items():
        if name not in variables:
            raise ValueError(f"unit_conversions names '{name}', which is not among the plotted variables {list(variables)}.")
        if not (isinstance(conversion, tuple) and len(conversion) == 2 and isinstance(conversion[1], str)):
            raise ValueError(f"unit_conversions['{name}'] must be a (factor, units_label) pair, "
                             f"such as (1e6, 'μM'); got {conversion!r}.")
    time_units = time_units or first.time_units

    number_of_rows = (len(variables) + columns - 1) // columns
    fig, axes = plt.subplots(number_of_rows, columns, figsize=figsize, constrained_layout=True, squeeze=False)
    axes_flat = axes.flatten()
    for index, name in enumerate(variables):
        ax = axes_flat[index]
        for result in results_list:
            time_axis = result.time_in(time_units)
            y_data = result[name]
            if name in unit_conversions:
                y_data = y_data * unit_conversions[name][0]
            if time_range is not None:
                mask = (time_axis >= time_range[0]) & (time_axis <= time_range[1])
                time_axis, y_data = time_axis[mask], y_data[mask]
            ax.plot(time_axis, y_data, label=result.condition_name, linewidth=2)
        ax.set_xlabel(f'Time ({time_units})')
        ax.set_ylabel(unit_conversions[name][1] if name in unit_conversions else first.units.get(name, ''))
        ax.set_title(first.descriptions.get(name, name))
        if log_scale in (True, 'loglog', 'logx'):
            ax.set_xscale('log')
        if log_scale in (True, 'loglog', 'logy'):
            ax.set_yscale('log')
        if time_range is not None:
            ax.set_xlim(time_range)
        if legend_style == 'individual':
            ax.legend()
        make_plot_look_good(ax)
    for index in range(len(variables), len(axes_flat)):
        axes_flat[index].axis('off')
    if legend_style == 'shared' and len(results_list) > 1:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=16, fontweight='bold')
    return fig, axes
