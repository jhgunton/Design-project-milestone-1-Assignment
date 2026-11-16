---
answer_type: text
created: 2026-11-10T08:49:55.855173
---
Does spacing matter? Yes. All regimens delivered three doses of 1.2 mg/kg, totaling 3.6 mg/kg, but changing the spacing substantially changed the living-cell population at day 63.

Interval with Fewest Living Cells on Day 63: Among the intervals tested, 21 days (listed as current value in the legend) produced the lowest day-63 living-cell fraction, leaving 4.766% of the initial population. Intervals of 2.1, 4.125, 8.101, 15.91, and 31.25 days left 46.36%, 32.75%, 16.07%, 5.473%, and 21.24%, respectively. So, of the intervals tested, 15.91 days produced the next fewest at 5.473% remaining.

How does the Blood Payload Peak and AUC Change with the Interval? The peak blood payload decreased as the interval increased from 1.549 ng/mL at 2.1 days to 0.6601 ng/mL at 21 days and 0.6125 ng/mL at 31.25 days, as seen in the top right graph. This makes sense as closely spaced doses produced more overlapping exposure and higher peaks. Conversely, AUC changed relatively little between intervals of 2.1 and 21 days, ranging from approximately 0.426–0.436 µg·h/mL. At 31.25 days, it decreased to 0.2777 µg·h/mL, but given the last dose in that regimen occurred on day 62.5, this is likely because much of of its subsequent exposure fell outside the 63-day observation window. It is likely that its total exposure would also be similar to the other intervals.

Why do much shorter and much longer intervals do worse? With short intervals, the cytoplasmic payload plot shows large early peaks followed by a long period of low payload. These high peaks provide diminishing increases in killing, and allows for substantial regrowth after treatment wears off (as seen in the living-cell plot). With long intervals, cytoplasmic payload falls low enough between injections to permit regrowth. The 31.25-day regimen also gives its last dose only half a day before assessment, leaving little time for that dose to kill cells. The 21-day regimen distributes effective killing more evenly across the observation period.

What Limits the Killing Rate when Cytoplasmic Payload is High? Given the killing rate follows a Monod expression, delta_max is the highest killing rate for P_c >> K_P. Conseuqently, even if cytoplasmic payload continues to increase (through shorter injection intervals for example), once P_c >> K_P, the killing rate won't increase proportionally with payload as delta_max is the max killing rate.

How Long a Dose Keeps Killing after It is Given? What the Tumor Does in Between? A dose continues contributing to killing while ADC persists and supplies intracellular payload. In comparing the cytoplasmic payload curve to the living cancer cell curve, you can see that this effect lasts for days to weeks rather than occurring only at injection.

The tumor population decreases while the killing rate exceeds the growth rate, i.e., delta > mu. As payload declines, killing slows. If delta < mu, surviving cells regrow, and with it the tumor. Thus, a dose has no fixed cutoff time: its duration of net tumor reduction depends on how long cytoplasmic payload remains sufficiently high to ensure delta > mu.
