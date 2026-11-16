---
answer_type: text
created: 2026-11-09T12:21:52.409068
---
I calculated the untreated receptor monomer and dimer steady state by setting the four receptor rate equations equal to zero and solving for the positive monomer concentration (via the quadratic formula, calculated using code). The resulting amounts were as shown on Page 2. 

I then used these values as initial conditions (along with N_0, calculated via the ratio of the volume of the tumor over the volume of one cell) to simulate 63 days without ADC or payload. The receptor time courses remained constant, consistent with steady state. Plotting the living-cell time course against N(t)=N_0e^(mu*t), where mu= ln(2)/(14 days in minutes). Given N_0=10^9 cells, the expected population at day 63 is 2.263 * 10^10 cells, or approximately 22.63 times the initial population. Agreement between the simulated and analytical growth curves for all times checks that the untreated system follows the specified 14-day doubling time.
