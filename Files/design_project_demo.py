"""Native model explorer, launched by design_project_helpers.

The notebook retains the model and its Python rate functions.
This process owns the desktop window and exchanges JSON with that local kernel.
"""

import json
import queue
import threading
import tkinter as tk
from tkinter import ttk
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class NotebookConnection:
    def __init__(self, url, token):
        self.url = url
        self.token = token

    def request(self, path, data=None, timeout=600):
        body = None if data is None else json.dumps(data).encode()
        request = Request(self.url + path, data=body, headers={
            'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json',
        })
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise ValueError(json.load(error).get('error', str(error))) from error
        except (URLError, OSError) as error:
            raise ValueError('The notebook connection is unavailable. Keep its kernel running and relaunch the explorer.') from error


class DosingWindow:
    """All Tk operations stay on the main thread; simulations run in a worker."""

    def __init__(self, root, connection, configuration):
        self.root = root
        self.connection = connection
        self.configuration = configuration
        self.reports = []
        self._generation = 0
        self._quiet = True
        self._busy = False
        self._closed = False
        self._responses = queue.Queue()
        self.parameter_vars = {}
        self.parameter_entries = {}
        self.controls = {}

        root.title('20.320 | ADC model explorer')
        root.geometry('1440x900')
        root.minsize(1120, 720)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(root, padding=12, width=410)
        sidebar.grid(row=0, column=0, sticky='nsew')
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(3, weight=1)

        ttk.Label(sidebar, text='Model explorer', font=('', 18, 'bold')).grid(row=0, sticky='w')
        ttk.Label(sidebar, text='Three equal doses on days 0, τ, and 2τ.\nAssessment on day 63.',
                  justify='left').grid(row=1, sticky='w', pady=(4, 10))
        ttk.Label(sidebar, text='Kinetic parameters', font=('', 12, 'bold')).grid(row=2, sticky='w')
        self._parameter_panel(sidebar)

        ttk.Label(sidebar, text='Values use the model units shown. Scientific notation is accepted.',
                  wraplength=380, foreground='#555555').grid(row=4, sticky='w', pady=6)
        dosing = ttk.LabelFrame(sidebar, text='Dosing schedule', padding=8)
        dosing.grid(row=5, sticky='ew', pady=(2, 8))
        dosing.columnconfigure(1, weight=1)
        specs = [
            ('dose', 'Each dose (mg/kg)', 0, configuration['maximum_dose']),
            ('dose_interval', 'Dose interval (days)', configuration['minimum_interval'], configuration['maximum_interval']),
        ]
        self.sliders = {}
        for row, (name, label, low, high) in enumerate(specs):
            variable = tk.StringVar(value=self._format(configuration[name]))
            self.controls[name] = variable
            ttk.Label(dosing, text=label).grid(row=row, column=0, sticky='w', pady=4)
            self.sliders[name] = ttk.Scale(dosing, from_=low, to=high, variable=variable,
                                            command=lambda value, key=name: self._slide(key, value))
            self.sliders[name].grid(row=row, column=1, sticky='ew', padx=6)
            ttk.Entry(dosing, textvariable=variable, width=7).grid(row=row, column=2)
            variable.trace_add('write', self._edited)

        sweep = ttk.LabelFrame(sidebar, text='Parameter sweep', padding=8)
        sweep.grid(row=6, sticky='ew')
        sweep.columnconfigure(1, weight=1)
        for row, (name, label) in enumerate([
            ('orders', 'Orders of magnitude +/-'), ('points', 'Points (2–101)'),
        ]):
            self.controls[name] = tk.StringVar(value=self._format(configuration[name]))
            ttk.Label(sweep, text=label).grid(row=row, column=0, sticky='w', pady=4)
            ttk.Entry(sweep, textvariable=self.controls[name], width=9).grid(row=row, column=1, sticky='e')
            self.controls[name].trace_add('write', self._edited)
        self.sweep_name = tk.StringVar(value='Choose a parameter to sweep…')
        self.sweep_options = {name: name for name in self.parameter_vars}
        self.sweep_options.update({'Dose size (mg/kg)': 'dose', 'Dose interval (days)': 'dose_interval'})
        self.sweep_menu = ttk.Combobox(sweep, textvariable=self.sweep_name, state='readonly',
                                       values=list(self.sweep_options), width=30)
        self.sweep_menu.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6, 4))
        self.sweep_button = ttk.Button(sweep, text='Sweep Parameter', command=self.start_sweep)
        self.sweep_button.grid(row=3, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        ttk.Label(sweep, text='Choose a parameter, then click Sweep Parameter to add a logarithmic sweep. '
                  'Dosing ranges stay within the slider bounds.',
                  wraplength=360, foreground='#555555').grid(row=4, column=0, columnspan=2, sticky='w')

        buttons = ttk.Frame(sidebar)
        buttons.grid(row=7, sticky='ew', pady=(12, 6))
        buttons.columnconfigure((0, 1), weight=1)
        self.update_button = ttk.Button(buttons, text='Update Plots', command=self.update_plots)
        self.update_button.grid(row=0, column=0, sticky='ew', padx=(0, 4))
        self.reset_button = ttk.Button(buttons, text='Reset Defaults', command=self.reset_defaults)
        self.reset_button.grid(row=0, column=1, sticky='ew', padx=(4, 0))
        ttk.Label(sidebar, text='Editing any numeric value clears the plots.\nReset restores the values present at launch.',
                  wraplength=380, foreground='#555555').grid(row=8, sticky='w', pady=4)

        main = ttk.Frame(root, padding=(0, 8, 10, 10))
        main.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        self.figure = Figure(figsize=(10, 6), constrained_layout=True)
        self.axes = self.figure.subplots(2, 2)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        toolbar = ttk.Frame(main)
        toolbar.grid(row=1, column=0, sticky='ew')
        NavigationToolbar2Tk(self.canvas, toolbar)
        self.results = ttk.Treeview(main, columns=('trace', 'cells', 'peak', 'auc'), show='headings', height=5)
        for column, heading, width in [
            ('trace', 'Curve', 330), ('cells', 'Day-63 cells (% initial)', 150),
            ('peak', 'Peak blood payload (ng/mL)', 175), ('auc', 'AUC (µg·h/mL)', 120),
        ]:
            self.results.heading(column, text=heading)
            self.results.column(column, width=width, minwidth=80, stretch=True)
        self.results.grid(row=2, column=0, sticky='ew', pady=6)
        scroll = ttk.Scrollbar(main, orient='vertical', command=self.results.yview)
        scroll.grid(row=2, column=1, sticky='ns')
        self.results.configure(yscrollcommand=scroll.set)
        self.status = tk.StringVar(value='Ready. Click Update Plots or Sweep Parameter.')
        self.status_label = ttk.Label(main, textvariable=self.status, wraplength=800)
        self.status_label.grid(row=3, column=0, sticky='w', pady=4)
        ttk.Label(main, text='Whole-blood exposure uses the assumed payload molecular weight. '
                  'Each variant starts at its own untreated receptor steady state. Keep the notebook kernel running.',
                  wraplength=850, foreground='#555555').grid(row=4, column=0, sticky='w')
        self._quiet = False
        self._draw()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self._poll_id = root.after(80, self._poll)

    @staticmethod
    def _format(value):
        return format(value, '.12g')

    def _parameter_panel(self, sidebar):
        frame = ttk.Frame(sidebar)
        frame.grid(row=3, sticky='nsew', pady=(6, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.parameter_canvas = tk.Canvas(frame, width=380, height=270, highlightthickness=0)
        self.parameter_canvas.grid(row=0, column=0, sticky='nsew')
        bar = ttk.Scrollbar(frame, orient='vertical', command=self.parameter_canvas.yview)
        bar.grid(row=0, column=1, sticky='ns')
        self.parameter_canvas.configure(yscrollcommand=bar.set)
        content = ttk.Frame(self.parameter_canvas)
        content.columnconfigure(1, weight=1)
        window = self.parameter_canvas.create_window((0, 0), window=content, anchor='nw')
        content.bind('<Configure>', lambda event: self.parameter_canvas.configure(scrollregion=self.parameter_canvas.bbox('all')))
        self.parameter_canvas.bind('<Configure>', lambda event: self.parameter_canvas.itemconfigure(window, width=event.width))
        content.bind('<MouseWheel>', self._scroll_parameters)
        for index, parameter in enumerate(self.configuration['parameters']):
            name = parameter['name']
            row = index * 2
            ttk.Label(content, text=name).grid(row=row, column=0, sticky='w', pady=(6, 0))
            variable = tk.StringVar(value=self._format(parameter['value']))
            self.parameter_vars[name] = variable
            entry = ttk.Entry(content, textvariable=variable, width=16)
            entry.grid(row=row, column=1, sticky='e', padx=(6, 0), pady=(6, 0))
            self.parameter_entries[name] = entry
            label = f"{parameter['units']} · {parameter['description']}"
            description = ttk.Label(content, text=label, wraplength=355, foreground='#555555')
            description.grid(row=row+1, column=0, columnspan=2, sticky='w', pady=(0, 5))
            for widget in [entry, description]:
                widget.bind('<MouseWheel>', self._scroll_parameters)
            variable.trace_add('write', self._edited)

    def _show_status(self, message, error=False):
        self.status.set(message)
        self.status_label.configure(foreground='#c00000' if error else '')

    def _scroll_parameters(self, event):
        self.parameter_canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')
        return 'break'

    def _slide(self, name, value):
        if self._quiet:
            return
        number = float(value)
        number = round(number, 2)
        text = self._format(number)
        if self.controls[name].get() != text:
            self.controls[name].set(text)

    def _edited(self, *_args):
        if self._quiet:
            return
        self._generation += 1
        self.reports.clear()
        self._draw()
        self._show_status('Values changed. Click Update Plots or Sweep Parameter.')

    def _request_values(self):
        import math

        try:
            parameters = {name: float(variable.get()) for name, variable in self.parameter_vars.items()}
            controls = {name: float(variable.get()) for name, variable in self.controls.items()}
        except ValueError as error:
            raise ValueError('Enter a number in every field. Scientific notation such as 1e-4 is accepted.') from error
        if not all(math.isfinite(value) for value in [*parameters.values(), *controls.values()]):
            raise ValueError('All values must be finite numbers.')
        if not self.configuration['minimum_interval'] <= controls['dose_interval'] <= self.configuration['maximum_interval']:
            raise ValueError(f"Dose interval must be from {self.configuration['minimum_interval']:g} "
                             f"to {self.configuration['maximum_interval']:g} days.")
        return {'parameters': parameters, **controls}

    def update_plots(self):
        self._start()

    def start_sweep(self):
        selection = self.sweep_name.get()
        if selection not in self.sweep_options:
            self._show_status('Choose a parameter from the sweep menu, then click Sweep Parameter.')
            return
        self._start(self.sweep_options[selection])

    def _start(self, sweep=None):
        if self._busy:
            self._show_status('A simulation is running. Wait for it to finish, then update or sweep.')
            return
        try:
            request = self._request_values()
        except ValueError as error:
            self._show_status(str(error), error=True)
            return
        request['include_baseline'] = not sweep or not self.reports
        if sweep:
            request['sweep'] = sweep
        self._busy = True
        self.update_button.state(['disabled'])
        self.reset_button.state(['disabled'])
        self.sweep_menu.state(['disabled'])
        self.sweep_button.state(['disabled'])
        self._show_status(f'Running {sweep} sweep…' if sweep else 'Updating plots…')
        generation = self._generation

        def work():
            try:
                result = self.connection.request('/evaluate', request)
                self._responses.put((generation, sweep, result, None))
            except Exception as error:
                self._responses.put((generation, sweep, None, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            generation, sweep, result, error = self._responses.get_nowait()
        except queue.Empty:
            pass
        else:
            self._busy = False
            self.update_button.state(['!disabled'])
            self.reset_button.state(['!disabled'])
            self.sweep_menu.state(['!disabled', 'readonly'])
            self.sweep_button.state(['!disabled'])
            if generation != self._generation:
                self._show_status('Values changed during the calculation. Click Update Plots to use the new values.')
            elif error:
                self._show_status(f'{"Sweep" if sweep else "Update"} failed. {error}', error=True)
            else:
                if not sweep:
                    self.reports.clear()
                self.reports.extend(result['reports'])
                self._draw()
                if sweep:
                    values = result['sweep_values']
                    self._show_status(f'Added {sweep} sweep: {len(values)} logarithmic points, '
                                    f'{values[0]:.4g} to {values[-1]:.4g}. '
                                    f'{len(self.reports)} curves shown; the current-values curve is retained.'
                                    + (' Range limited to the dosing bounds.' if result.get('range_limited') else ''))
                else:
                    report = self.reports[0]
                    outcome = '95% target met' if report['target_met'] else '95% target not met'
                    dose = float(self.controls['dose'].get())
                    self._show_status(f'{outcome} · Reduction {report["reduction_percent"]:.5f}% · '
                                    f'{dose:g} mg/kg per dose, {3*dose:g} mg/kg total.')
        if not self._closed:
            self._poll_id = self.root.after(80, self._poll)

    def _draw(self):
        panels = [
            ('adc_ug_per_mL', 'Circulating ADC', 'Whole blood (µg/mL)'),
            ('payload_ng_per_mL', 'Circulating released payload', 'Whole blood (ng/mL)'),
            ('cytoplasmic_uM', 'Cytoplasmic payload', 'Intracellular concentration (µM)'),
            ('cell_fraction', 'Living cancer cells', 'Living cells / initial'),
        ]
        from matplotlib import colormaps
        colors = colormaps['tab10']
        for legend in list(self.figure.legends):
            legend.remove()
        for axis, (key, title, label) in zip(self.axes.flat, panels):
            axis.clear()
            for index, report in enumerate(self.reports):
                color = '#222222' if report['label'] == 'Current values' else colors((index - 1) % 10)
                style = '-' if index < 11 else '--'
                axis.plot(report['days'], report[key], label=report['label'], color=color,
                          linewidth=1.7, linestyle=style)
            schedules = {tuple(report['dose_days']) for report in self.reports if 'dose_days' in report}
            try:
                interval = float(self.controls['dose_interval'].get())
                # Shared dose markers apply only when all curves use the same schedule.
                if len(schedules) <= 1:
                    days = next(iter(schedules), (0, interval, 2 * interval))
                    for day in days[1:]:
                        axis.axvline(day, color='.75', linestyle=':', linewidth=.7)
            except ValueError:
                pass
            axis.set(title=title, xlabel='Time (days)', ylabel=label, xlim=(0, self.configuration['assessment_day']))
            axis.grid(alpha=.2)
        self.axes[1, 1].set_yscale('log')
        self.axes[1, 1].axhline(.05, color='.4', linestyle='--', linewidth=1, label='95% reduction target')
        self.axes[1, 1].set_ylim((.01, 2) if not self.reports else (None, None))
        if self.reports:
            handles, labels = self.axes[0, 0].get_legend_handles_labels()
            if len(handles) > 12:
                from matplotlib.lines import Line2D
                extra = len(handles) - 12
                handles = handles[:12] + [Line2D([], [], linestyle='none')]
                labels = labels[:12] + [f'{extra} more curves listed in the table']
            self.figure.legend(handles, labels, loc='outside lower center', ncols=2, fontsize=7)
        self.results.delete(*self.results.get_children())
        for report in self.reports:
            self.results.insert('', 'end', values=(report['label'], f'{100*report["final_fraction"]:.4g}',
                                f'{report["payload_peak_ng_per_mL"]:.4g}', f'{report["payload_auc_ug_h_per_mL"]:.4g}'))
        self.canvas.draw_idle()

    def reset_defaults(self):
        self._quiet = True
        try:
            for parameter in self.configuration['parameters']:
                self.parameter_vars[parameter['name']].set(self._format(parameter['value']))
            for name, variable in self.controls.items():
                variable.set(self._format(self.configuration[name]))
            self.sweep_name.set('Choose a parameter to sweep…')
        finally:
            self._quiet = False
        self._edited()
        self.update_plots()

    def bring_to_front(self):
        """A window launched from the notebook kernel otherwise opens behind VS Code on macOS."""
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        self.root.after(350, lambda: self.root.attributes('-topmost', False))

    def close(self):
        self._closed = True
        self.root.after_cancel(self._poll_id)
        self.root.destroy()

def main():
    import sys

    connection = NotebookConnection(sys.argv[1], sys.argv[2])
    configuration = connection.request('/configuration')
    root = tk.Tk()
    window = DosingWindow(root, connection, configuration)
    window.bring_to_front()
    root.after(100, window.update_plots)
    try:
        root.mainloop()
    finally:
        try:
            connection.request('/close', {}, timeout=2)
        except Exception:
            pass


if __name__ == '__main__':
    main()
