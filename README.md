# Settlement Feasibility & Fee Engine — Take-home

Welcome, and thanks for taking the time. The full problem is in
[`ASSIGNMENT.md`](./ASSIGNMENT.md). This README is just orientation.

## The task in one line

Given a client's escrow account, a settlement offer, and a creditor's rules,
decide whether the offer is affordable (and schedule it, collecting our fee as
early as allowed) or — if not — compute the minimum extra funding needed.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
hiring_takehome/
├── ASSIGNMENT.md            # full specification — read this
├── feasibility/
│   ├── models.py            # data models, JSON loaders, date/EOM helpers (provided)
│   └── engine.py            # >>> implement evaluate_offer here <<< (+ Result shape)
├── cases/                   # four example cases (client.json / offer.json / creditor_rules.json)
│   ├── case1_feasible_even
│   ├── case2_infeasible_minima
│   ├── case3_balloon
│   └── case4_tiers
├── tests/
│   ├── test_smoke.py        # scaffolding sanity tests (pass out of the box)
│   └── test_cases.py        # example expectations — make these pass, then add your own
├── run.py                   # python run.py cases/<case>
└── requirements.txt
```

## Run

```bash
# evaluate a single case (prints the Result as JSON)
python run.py cases/case1_feasible_even

# tests
pytest -q
```

Out of the box, `tests/test_smoke.py` passes and `tests/test_cases.py` fails —
the latter is your target. Go beyond those four cases with your own tests.

## What to submit

Your implementation, your tests, and a short README section describing:
- your approach and the alternatives you considered,
- **your interpretation of the payment shapes** (even / staircase / balloon — we
  left these loosely defined on purpose),
- assumptions you made, and known edge cases / limitations.

Budget ~5–6 hours. Prefer a correct, well-tested core over breadth. When in
doubt, write down your assumption and keep going.

---

# Implementation notes

Everything below is my write-up: the model, the shape interpretation, the
assumptions I had to make, and what I know is weak.

## Layout of the solution

| File | Contents |
|---|---|
| `feasibility/money.py` | round-half-up, and percentage arithmetic that does not drift through floats |
| `feasibility/solver.py` | the model: cadence, cash caps, payment shapes, fee placement, Part 2 minima |
| `feasibility/engine.py` | `evaluate_offer` and the (unchanged) output dataclasses |
| `tests/test_engine.py` | the full suite, including an exhaustive cross-check and a fuzz |
| `tests/test_conformance.py` | an independent re-reading of the spec, brute-forced against the engine |

`feasibility/models.py` is the provided scaffolding with three small changes,
all flagged under *Assumptions* below.

## Approach

**The whole problem collapses into a prefix-budget constraint.** Our creditor
payments, program fee and bank fees are the only movable entries; everything
else in the ledger is fixed. So I fold the committed ledger into a single cash
curve `C(t)` — the balance the account *would* have if we scheduled nothing —
and then the entire feasibility question becomes, for each cadence date `i`:

```
(everything we have debited through date i)  ≤  cap[i]
```

where `cap[i]` is the *minimum* of `C(t)` over the window from cadence date `i`
up to just before date `i+1`. Taking the window minimum rather than the value at
`d_i` is what enforces "balance ≥ 0 at **every** date" and not merely on the
dates we touch — a committed debit landing between two cadence dates is caught
here.

Two conveniences fall out of this:

- Because credits precede debits on a date, and because a date is non-negative
  only if it is non-negative after *all* of its debits, same-day ordering
  between our debits and someone else's does not matter. The two net.
- Fee collected at date `i` is still sitting outside the account at every later
  date, while creditor payments and bank fees keep accumulating. So the amount
  of fee we may hold by date `i` is bounded not by that date alone but by
  **every** later one:

  ```
  room[i] = min over j ≥ i of ( cap[j] − cumulative_payments[j] − cumulative_bank_fees[j] )
  ```

  Being a suffix minimum, `room` is **non-decreasing**, which is what licenses
  the one-pass fee greedy below. The suffix minimum has to be taken over the
  *whole difference*: folding it into `cap` alone is wrong, because the
  subtrahend grows with `j` too. (That was a real bug — see *Testing*.)

**Given a payment vector, fee placement is then a greedy with no lookahead.**
The fee total is fixed, so *when* we take it does not change the final
cumulative outflow — it only shifts intermediate prefixes. Walk the cadence
dates and take `min(fee remaining, room[i] − collected)` at each. Since `room`
is non-decreasing, taking the maximum now can never strand the remainder later;
feasibility reduces to the single global check that everything fits by the
horizon. This is provably lexicographically optimal, not a heuristic.

**So the only real search is over the payment vector.** With the greedy folded
in, the collected-fee-to-date at cadence date `i` is exactly

```
F[i] = min(total_fee, room[i])
```

which means *maximising the fee lexicographically is identical to minimising
cumulative (payments + bank fees) lexicographically*. That single quantity is
the objective, and it explains the shapes without any special-casing: paying the
creditor the least the rules allow, as late as the rules allow, is what frees the
early dollars for our fee. It also prices the count `k` correctly — a larger `k`
means smaller early payments but one more bank fee each month, and the objective
weighs both in the same currency.

I search every `k` from 1 to `min(max_payments, max_terms, #cadence dates)`,
generate the candidate vectors for that `k`, keep the cash-feasible ones, and
take the lexicographic best. Ties (which happen whenever the program fee is
zero, as in case 3) break toward deferring creditor money, then toward fewer
payments.

### Alternatives I considered

- **CP-SAT / an ILP** (`ortools`). This is a genuinely small integer program and
  a solver would express the constraints almost verbatim. I skipped it: it adds
  a heavy dependency for a problem whose structure is transparent enough to
  solve exactly in closed form, and a lexicographic objective over ~12 dates is
  awkward to express (it needs either weights big enough to be fragile, or an
  iterated solve per date). Being able to *prove* the greedy optimal was worth
  more than the generality.
- **DP over (position, cents spent).** Correct but the state space is the offer
  total in cents — millions of states for no benefit, since the optimal
  structure is characterisable directly.
- **Fixing the shape per flag and skipping the `k` search** (e.g. "even → use
  max k"). This is wrong whenever `bank_fee_cents > 0`: more payments buy
  smaller early payments at the cost of another bank fee, and which side wins
  depends on the numbers. Case 1 happens to prefer the largest `k`; that is an
  outcome, not a rule.
- **Enumerating every non-decreasing payment vector.** Exponential, but it is
  exactly what `tests/test_engine.py` does on small inputs to verify the pruned
  search — see *Testing* below.

## Payment shapes — my interpretation

This is the deliberately open-ended part, so here is precisely what I chose.

### Floors, first

All three shapes sit on a per-position floor that combines the three sources in
constraint 4. The **token-pay rule becomes a positional floor** because payments
are non-decreasing: every payment equal to the base minimum therefore forms a
prefix of the schedule, so a payment past position `max_token_pays` cannot be at
the base and must strictly exceed it — I read "must exceed" as `base + 1` cent.
Tiers layer on top with a max. The result is cumulative-maximum'd, since a
non-decreasing sequence that respects each floor respects all earlier ones.

### What lands in `pay_shape_used`

The shape is reported off the vector that came out, not off the flags that went
in — a creditor who *permits* ballooning does not necessarily get one.

- **"even"** — the payments are as-equal-as-possible, i.e. equal to their own
  `even_split`. Note this is tested against `even_split` rather than "all
  identical". §7 defines evenness as "as equal as possible", so `[8333, 8334]`
  is even in exactly the way `[8333, 8333]` is; testing for a single distinct
  value would make the reported shape hinge on whether `k` happens to divide
  `offer_total`, flipping the same schedule from "even" at $100.00 to
  "staircase" at $100.01.
- **"balloon"** — the final payment jumps by **two cents or more**. An
  as-equal-as-possible level spans at most one cent, so that gap is exactly what
  separates §2's "final payment absorbing the entire remaining balance" from the
  `+1`-cent remainder of an evenly-split level.
- **"staircase"** — anything else: a genuine step up, with the last level shared
  by at least two payments.

### `even_pays = true` → **"even"**

All payments equal, with the remainder cents on the latest payments per
constraint 7. The vector is forced once `k` is chosen, so the only decision is
`k`, taken by the objective. `max_segments` is ignored, as specified.

### `is_ballooning_allowed = true` → **"balloon"**

Every payment but the last sits exactly on its floor; the final payment absorbs
the entire remainder. **How token pays and tiers interact with the balloon:** they
constrain the small early payments exactly as they always do — the balloon does
not exempt them — and they apply to the balloon payment itself too, which is
vacuous because the balloon is by construction the largest payment and so clears
its own floor and any tier. In case 3 that gives five token pays at the $25
minimum and a $175 balloon; the token cap of 6 is what permits five of them, and
a cap of, say, 2 would force payments 3–5 up to $25.01 before the balloon.

`max_segments` is ignored here (per §4), which is the only reason the balloon can
exist at all — floors + a distinct final level is often three levels or more.

This vector is worth calling out: it simultaneously minimises *every* prefix
sum, so it dominates all other vectors under the objective. If the balloon does
not fit the cash for a given `k`, nothing else for that `k` will either, so the
solver need not consider anything else. The brute-force test confirms this.

### Neither flag → **"staircase"**

A staircase is a partition of the payments into at most `max_segments`
consecutive blocks, each block flat. Two rules pin down where the steps go:

1. **A non-final block sits at the highest floor it covers.** Raising it above
   that only moves money earlier, which the objective never wants.
2. **The final block absorbs the remainder**, split as equally as possible with
   the remainder cents on its latest payments — the same rule constraint 7 gives
   for `even_pays`.

So the shape is "hug the floors for as long as the segment budget allows, then
step up once to whatever clears the balance". Case 4 is the clean illustration:
six token pays at $25, then the `[7, $50]` tier forces a step, and with
`max_segments = 2` the second level has to carry all $450 of the remainder →
`[$25 × 6, $75 × 6]`. Starting the step later (which the objective would prefer)
would need a third level, so the cap is what picks the answer.

**Two sub-interpretations inside this, both of which do real work:**

- **The `+1`-cent remainder of a level is not a level of its own.** Otherwise
  "as equal as possible" would itself consume a segment and `max_segments = 1`
  would demand that `k` divide the offer total exactly — which would make case 1
  style inputs infeasible for arbitrary reasons of divisibility. A level and its
  one-cent rounding partner count as one segment.
- **A staircase's final level must be shared by at least two payments** (when
  `k ≥ 2`). Without this the objective always produces a lone, huge final
  payment — which is a balloon, and constraint 8 says a balloon requires the
  flag. The segment cap alone does *not* prevent this: `[$25, $25, $25, $450]` is
  only two levels. So "balloon" is the shape whose last level has exactly one
  payment, and forbidding that is what makes the flag meaningful. A visible
  consequence: with `k = 2` and ballooning off, the two payments must be
  as-equal-as-possible, since any other split leaves the last level alone.

### Why the pruned search is still exact

Enumerating every block structure is `C(k−1, s−1)` summed over `s ≤
max_segments`, which is fine at `k = 12` but not at `k = 60`. Two reductions cut
it to a handful of candidates without losing the optimum:

- A non-final block that ends mid-plateau can be extended to the end of that
  floor plateau at no cost, and doing so hands the positions it swallows a
  *lower* value than the next block would have. So non-final block ends only
  need to be considered at floor plateau ends.
- The one exception is the last non-final boundary, which is also capped at
  `k−2` (the latest start the final block may have, given it needs two
  payments).

That leaves `O(#plateaus + 1)` candidate positions. `tests/test_engine.py`
verifies the reduction against exhaustive enumeration over 195 parameter
combinations rather than asking you to take the argument on faith.

## Part 2 — minimum additional funds

Both minima are monotone: injecting money only lifts the cash curve, so the set
of feasible schedules only grows. I bracket by doubling and then bisect, which
needs ~2·log₂(L) solver calls and lands on the exact cent.

- **Lump sum** goes on `as_of_date + 1 day`, the earliest date we are allowed to
  modify. An earlier lump is available at every date a later one is, so the
  smallest `L` is always attained at the earliest date — there is no trade-off
  to explore. For case 2 that is 2026-01-01.
- **Monthly increment** is modelled as an extra credit on each future draft
  date, which sidesteps cloning the client. `N` counts every draft dated after
  `as_of_date`, including ones that arrive too late to help.

Case 2 shows why the two totals differ, exactly as the assignment predicts: the
last cadence date is Apr 30, so the May 1 draft is dead weight. The lump needs
$100; the increment needs $25 across 5 drafts = $125, because only 4 of those 5
drafts land in time to do anything.

Guardrails compare with `>`, so an amount landing exactly on the cap passes.

## Assumptions

1. **`creditor_balance_cents` vs `current_balance_cents`.** ASSIGNMENT.md §3 says
   the offer field was renamed, but `models.py` and all four `offer.json` files
   still use `current_balance_cents`. I took the code and data as ground truth:
   `load_offer` accepts **either** key, and `Offer.creditor_balance_cents` is a
   property aliasing the field, so both spellings work.
2. **`round()` in the provided `models.py` was half-to-even**, which contradicts
   §3. I replaced both helpers with the half-up implementation in `money.py`.
   Percentages also go through `Decimal(str(pct))` so that `0.07 × 114500` is
   8015 and not 8015.000000000001.
3. **A creditor payment is at least one cent.** With `min_payment_cents = 0` a
   schedule could put zeros on the leading cadence dates and effectively start
   later while claiming to start at `first_payment_date`, which constraint 1
   forbids — and it would draw a bank fee for moving no money.
4. **"The credits in the ledger are the drafts."** I take the ledger literally
   rather than re-deriving drafts from `draft_amount_cents` / `draft_day`; the
   two agree in all four cases. Ledger entries dated on or before `as_of_date`
   are skipped, since they are already inside `current_balance_cents`.
5. **Fee-collection dates are cadence dates.** §6 says the fee is "collected
   across cadence dates", so I never place it off-cadence, including on the
   horizon date when that is not itself a cadence date.
6. **Emitted rows** are the dates that carry a creditor payment plus any
   fee-only date. Cadence dates where nothing happens are omitted.
7. **Ordering among same-day debits is unspecified and does not matter**: the
   balance is checked after all of them, which is the strictest reading.
8. When the fee is zero the objective is degenerate, so the tie-break (defer
   creditor money) decides the shape. This is what produces the balloon in case
   3 rather than any appeal to the flag.

## Known edge cases and limitations

- **Structural infeasibility is reported distinctly.** When no schedule exists
  at *any* funding level — floors that cannot sum to the offer total, a first
  payment date past the horizon, a committed ledger that already goes negative
  before the first cadence date — the funding search cannot converge. I report
  `amount_cents: 0`, `within_guardrail: false` and a reason naming the cause,
  rather than a misleading huge number. This is a shape the assignment does not
  specify.
- **A negative `current_balance_cents`** is nonsense for an escrow account, but
  it is reported honestly rather than blamed on the rules. Nothing dated after
  `as_of_date` can repair a balance that already went negative on it, so the
  offer is infeasible at every funding level — and because that is genuinely a
  *cash* shortfall, the reason says so instead of pointing at dates or floors.
  The alternative reading (simulate only dates strictly after `as_of_date`, and
  let a lump lift them) is defensible too; I took the conservative one, since
  promising a schedule on an already-overdrawn account seems worse than
  declining one.
- **`max_segments = 1` with a stepped floor** is often genuinely infeasible: one
  level cannot straddle a tier. The solver reports infeasible and Part 2 then
  correctly finds that money does not help.
- **`k = 2` with ballooning off** forces two near-equal payments, per the
  final-level rule above. Defensible, but it is an interpretation, not a
  deduction.
- **`max_terms` and `max_payments`** are treated as redundant (`k ≤ min(...)`),
  as the assignment's author note anticipates.
- **An `offer_total` of zero** (a settlement percentage or creditor balance of
  zero) is reported infeasible, because constraint 1 still demands `k ≥ 1` and
  assumption 3 puts every payment at a cent or more. A degenerate input rather
  than a case I think is worth special-casing.
- **A `first_payment_date` before `as_of_date`** would schedule payments into
  the past. I take constraint 1 literally and start the cadence there anyway;
  those dates then see only `current_balance_cents`, since the ledger entries
  behind them are already baked in. Contradictory input, flagged rather than
  silently repaired.
- **Performance.** The search is `O(k_max × candidates × #cadence dates)` with a
  handful of candidates, so it is instant at realistic sizes; the whole suite of
  300 tests runs in about two seconds. Very large `max_segments` combined with many
  distinct floor tiers would grow the candidate set combinatorially, which I have
  not guarded beyond the plateau reduction.
- **Not modelled:** partial/failed drafts, mid-program cancellation, interest or
  fees accruing on the creditor balance, and multiple concurrent settlements
  competing for the same SDA (other settlements appear only as fixed debits).

## Testing

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest -q          # 300 passed
```

`tests/test_cases.py` (the provided bar) and `tests/test_smoke.py` both pass
unmodified. `tests/test_engine.py` adds the real coverage:

- **An independent validator.** `assert_valid_schedule` re-implements all ten
  hard constraints of §5 from scratch — re-simulating the ledger date by date,
  credits before debits, rather than trusting any number the engine reported —
  and re-derives the floors independently. **Every** feasible result in the
  suite, including all fuzz results, is checked through it.
- **The §10 checklist**, each as a named test: the three shapes; token-pay and
  tier floors (including `max_token_pays = 0` and overlapping tiers); the
  `max_segments` cap and its monotonicity; exact sum; same-day ordering (a case
  that is feasible *only* if the credit lands first); a balance that touches
  exactly $0 and the one-cent-short case that must not; the horizon limit and
  cadence clamping; no fee before the first payment, fee-only dates carrying no
  bank fee, and a fee that cannot be collected in time; and both Part 2 minima
  with their guardrails.
- **The §6 worked micro-example**, encoded directly. We collect the full $50 fee
  on the first date as the assignment describes, but pay `[$25, $112.50,
  $112.50]` rather than its illustrative `[$50, $100, $100]`. Both collect the
  whole fee at the earliest possible moment, so they tie on the objective and
  the tie-break decides — and §6's own wording ("keep creditor payments as low
  as the rules allow early on") points at the lower first payment.
- **Dates that are easy to get wrong**: a preserved-day cadence clamping through
  February without degenerating into an end-of-month cadence, a leap day, a
  committed debit landing *between* two cadence dates (which forces the fee
  greedy to hold cash back), and `k` being chosen by the objective rather than
  maximised when the bank fee makes extra payments too expensive.
- **An exhaustive cross-check** (195 combinations): for small totals, brute-force
  every legal non-decreasing payment vector and confirm the pruned search finds
  the same optimum. This is what verifies the block-structure reductions and the
  claim that the balloon dominates. It caught two real defects during
  development — an over-strict floor check that rejected legal schedules whose
  rounding remainder landed on a tier step-up, and the zero-payment loophole in
  assumption 3.
- **Two seeded fuzz tests** over 300 random scenarios: every feasible result must
  pass the validator, and every infeasible one must report minima that are
  exactly minimal (`L` works, `L−1` does not; likewise `X`).

### `tests/test_conformance.py` — an independent second opinion

Everything above tests the engine against *my* model of the problem, so a
misreading of the spec would be invisible to it. This file is the control: it
re-derives ASSIGNMENT.md from scratch — rounding, cadence, floors, shape
legality, the ledger walk — and imports nothing from `feasibility/solver.py`.
On deliberately tiny inputs (single-digit cents; the arithmetic is identical,
the search space is not) it brute-forces **every** valid `(payment vector, fee
split)` pair and requires the engine to agree on the verdict and to be at least
as good on the objective. Part 2's minima are re-checked the same way: `L` must
work against the brute-force oracle and `L−1` must not.

**It found two real bugs that the suite above did not**, neither of them visible
in the four provided cases, and both of which all 256 tests above passed
straight through:

1. **The fee greedy stranded a later creditor payment.** It computed headroom as
   `suffix_min(cap)[i] − cumulative_payments[i]`, when the suffix minimum has to
   be taken over the whole difference (see *Approach*). Where the fee and a
   *later* payment competed for the same cash, the greedy took the fee first and
   then reported the entire offer **infeasible** — a false negative, with
   correspondingly inflated Part 2 minima.
2. **An indivisible even schedule was misreported.** `classify_shape` tested
   `len(set(payments)) == 1`, so it only recognised evenness when `k` divided
   `offer_total` exactly. `[8333, 8334]` — §7's "as equal as possible" — came
   back as `"staircase"`, or as `"balloon"` where the creditor allowed
   ballooning, even though a one-cent rounding remainder is not a final payment
   absorbing the remaining balance.

Both are now pinned by named regression tests, and the validator checks the
reported `pay_shape_used` against an independently derived shape.

The oracle also confirms the **staircase pruning is exact**, which is the
argument in *Why the pruned search is still exact* that most deserved checking.
Over 101,376 configurations (`k` up to 8, four base minimums, four token caps,
six tier layouts including overlapping tiers, segment caps 1–4) every generated
candidate is legal, and every legal vector is dominated at *every* prefix by
some generated candidate — which is precisely what licenses discarding the rest.

Beyond the suite, the oracle was run over 4,000 randomized scenarios (mid-month,
end-of-month and omitted cadences; committed debits straddling cadence dates; an
`as_of_date` mid-ledger; overlapping tiers) with no disagreement.

Finally, a battery of **fifteen degenerate inputs the spec never contemplates** —
an unsorted ledger, duplicate ledger dates, an entry landing exactly on
`as_of_date`, a first payment date on the horizon and one before `as_of_date`,
`max_terms = 0`, `max_segments = 0`, a zero offer total, a fee equal to the whole
balance, an empty ledger, a leap-day cadence, a negative opening balance — each
of which must neither crash nor emit an invalid schedule, and must serialize.
That battery is what turned up the misleading reason string above.

### §10 checklist → where it is covered

| §10 requirement | Test |
|---|---|
| even / staircase / balloon shapes | `test_even_shape_distributes_remainder_onto_latest_payments`, `test_staircase_never_ends_in_a_lone_balloon_payment`, `test_balloon_defers_everything_to_the_final_payment`, `test_an_indivisible_even_schedule_is_reported_even` |
| token-pay and tier floors | `test_token_pay_cap_forces_later_payments_above_the_base`, `test_zero_token_pays_puts_every_payment_strictly_above_the_base`, `test_tier_floor_applies_from_its_payment_number_onward`, `test_overlapping_tiers_take_the_strictest` |
| `max_segments` cap | `test_single_segment_forces_one_level`, `test_segment_cap_binds_against_the_floor_staircase`, `test_more_segments_never_produce_a_worse_prefix` |
| exact sum | `test_payments_sum_exactly_to_the_rounded_offer_total` (incl. totals that do not divide by `k`, and totals whose last cent depends on half-up rounding) |
| date-by-date simulation | `test_same_day_credit_is_applied_before_the_debit`, `test_committed_debits_are_respected_not_modified`, `test_fee_is_held_back_for_a_debit_between_cadence_dates` |
| balance hits exactly $0 | `test_balance_may_touch_exactly_zero_but_not_go_below`, `test_one_cent_short_is_infeasible` |
| horizon limit | `test_cadence_stops_at_the_horizon`, `test_first_payment_date_past_the_horizon_is_infeasible`, `test_fee_that_cannot_be_collected_by_the_horizon_is_infeasible` |
| fee compliance | `test_no_fee_before_the_first_creditor_payment`, `test_fee_only_dates_carry_no_bank_fee`, `test_bank_fee_charged_once_per_payment_date` |
| both Part 2 minima | `test_case2_minima_match_and_are_minimal`, `test_reported_minima_are_exactly_minimal`, `test_guardrails_reject_oversized_funding`, `test_guardrail_accepts_an_amount_exactly_at_the_cap` |

Every feasible result produced anywhere in the suite additionally goes through
one of the two independent validators, so exact sum, the floors, the bank-fee
rule, the fee-timing rules and the date-by-date balance are re-checked on each
one, not only in the named tests above.
