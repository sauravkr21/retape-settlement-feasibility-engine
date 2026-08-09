"""Conformance tests built on an *independent* brute-force oracle.

``tests/test_engine.py`` tests the engine against my model of the problem. This
file does the opposite: it re-derives the spec from ASSIGNMENT.md alone —
rounding, cadence, floors, shape legality, the ledger simulation — and then
brute-forces **every** valid (payment vector, fee split) pair on deliberately
tiny inputs. Nothing here imports ``feasibility.solver``, so agreement between
the two is evidence rather than a shared assumption.

Amounts in the randomized tests are single-digit cents. The arithmetic is
identical to realistic magnitudes (it is all integer cents), but the search
space stays small enough to enumerate exhaustively.
"""

from __future__ import annotations

import itertools
import random
from calendar import monthrange
from datetime import date
from decimal import ROUND_HALF_UP, Decimal

import pytest

from feasibility.engine import evaluate_offer
from feasibility.models import Client, CreditorRules, LedgerEntry, Offer, load_case


# ---------------------------------------------------------------------------
# The oracle: ASSIGNMENT.md, re-implemented from scratch
# ---------------------------------------------------------------------------

def rnd(pct, cents: int) -> int:
    """§3: round-half-up(pct * cents)."""
    return int((Decimal(str(pct)) * Decimal(int(cents))).quantize(
        Decimal(1), rounding=ROUND_HALF_UP))


def _last_day(y: int, m: int) -> int:
    return monthrange(y, m)[1]


def cadence(client: Client, offer: Offer) -> list[date]:
    """§3: monthly cadence from first_payment_date through the horizon."""
    if offer.first_payment_date is None:
        f = client.first_draft_date
        start = date(f.year, f.month, _last_day(f.year, f.month))
    else:
        start = offer.first_payment_date
    true_eom = start.day == _last_day(start.year, start.month)
    out: list[date] = []
    for i in range(400):
        total = start.year * 12 + start.month - 1 + i
        y, m = divmod(total, 12)
        m += 1
        day = _last_day(y, m) if true_eom else min(start.day, _last_day(y, m))
        cur = date(y, m, day)
        if cur > client.last_draft_date:
            break
        out.append(cur)
    return out


def floors_ok(payments: list[int], rules: CreditorRules) -> bool:
    """§5.4, read straight off the spec wording."""
    base = rules.min_payment_cents
    if any(p < base for p in payments):
        return False
    # "at most max_token_pays payments may sit AT the base minimum"
    if sum(1 for p in payments if p == base) > rules.max_token_pays:
        return False
    for i, p in enumerate(payments, start=1):
        for frm, cents in rules.min_payment_tiers:
            if i >= frm and p < cents:
                return False
    return True


def even_split(total: int, n: int) -> list[int]:
    q, r = divmod(total, n)
    return [q] * (n - r) + [q + 1] * r


def segments(payments: list[int]) -> int:
    """The number of distinct payment levels this vector uses (minimised).

    A level is a *flat* run. The single exception is the last level, which has
    to absorb whatever remains and so may be split "as equal as possible" per
    §5.7 -- ``[3, 4]`` for a 7c level over two payments is ONE level, not two.
    Without that exception ``max_segments = 1`` would demand that ``k`` divide
    ``offer_total`` exactly. The exception is deliberately *not* extended to
    non-final levels: those sit on a floor and never need a rounding remainder,
    and allowing it there would make ``[1, 2, 3, 4]`` a two-level staircase.
    """
    k = len(payments)
    # flat_blocks[i] = minimal flat blocks covering payments[:i] = number of runs
    flat_blocks = [0] * (k + 1)
    for i in range(1, k + 1):
        flat_blocks[i] = flat_blocks[i - 1] + (
            1 if i == 1 or payments[i - 1] != payments[i - 2] else 0
        )
    best = k  # worst case: every payment its own level
    for i in range(k):
        tail = payments[i:]
        if tail == even_split(sum(tail), len(tail)):
            best = min(best, flat_blocks[i] + 1)
    return best


def is_balloon(payments: list[int]) -> bool:
    """§2: small payments early, one large final payment absorbing the rest.

    A final level of length >= 2 is possible exactly when the last two payments
    are within a rounding cent of each other (an "as equal as possible" tail has
    max - min <= 1, and that only gets harder for longer tails). So a balloon is
    precisely a last payment that jumps by more than one cent -- a genuine step
    up, not the +1c remainder of an evenly-split level.
    """
    return len(payments) >= 2 and payments[-1] - payments[-2] >= 2


def shape_ok(payments: list[int], rules: CreditorRules) -> bool:
    """§5.7, §5.8, §5.9."""
    if rules.even_pays:
        return payments == even_split(sum(payments), len(payments))
    if rules.is_ballooning_allowed:
        return True  # segment cap ignored; a balloon is permitted
    if is_balloon(payments):
        return False  # §5.8: a balloon needs the flag
    return segments(payments) <= max(1, rules.max_segments)


def payment_vectors(rules: CreditorRules, k: int, total: int) -> list[list[int]]:
    """Every legal non-decreasing vector of length k summing to total."""
    out: list[list[int]] = []

    def rec(prefix: list[int], remaining: int, lo: int) -> None:
        i = len(prefix)
        if i == k:
            if remaining == 0 and floors_ok(prefix, rules) and shape_ok(prefix, rules):
                out.append(list(prefix))
            return
        left = k - i
        for v in range(lo, remaining // left + 1):
            rec(prefix + [v], remaining - v, v)

    rec([], total, max(1, rules.min_payment_cents))
    return out


def simulate(client, dates, payments, fees, bank_fee, extra=()):
    """§5.10: date-by-date walk, all credits before all debits."""
    credits: dict[date, int] = {}
    debits: dict[date, int] = {}
    for e in list(client.ledger) + list(extra):
        if e.date <= client.as_of_date:
            continue  # already inside current_balance_cents
        bucket = credits if e.type == "credit" else debits
        bucket[e.date] = bucket.get(e.date, 0) + e.amount_cents
    for i, d in enumerate(dates):
        if payments[i]:
            debits[d] = debits.get(d, 0) + payments[i] + bank_fee
        if fees[i]:
            debits[d] = debits.get(d, 0) + fees[i]

    bal = client.current_balance_cents
    if bal < 0:
        return None
    out: dict[date, int] = {}
    for d in sorted(set(credits) | set(debits)):
        bal += credits.get(d, 0)
        bal -= debits.get(d, 0)
        if bal < 0:
            return None
        out[d] = bal
    return out


def fee_splits(total: int, m: int):
    """Every way to spread `total` cents across m cadence dates."""
    for cuts in itertools.combinations_with_replacement(range(m), total):
        parts = [0] * m
        for c in cuts:
            parts[c] += 1
        yield parts


def best_schedule(client, offer, rules, extra=(), first_only=False):
    """Exhaustive search; returns the fee-front-loaded optimum.

    §6's objective is "collect the program fee as early as possible", i.e.
    maximise the vector of cumulative fee collected, lexicographically.
    """
    dates = cadence(client, offer)
    if not dates:
        return None
    total = rnd(offer.settlement_pct, offer.creditor_balance_cents)
    fee_total = rnd(rules.program_fee_pct, offer.original_balance_cents)
    m = len(dates)
    best = None
    for k in range(1, min(rules.max_payments, rules.max_terms, m) + 1):
        for pv in payment_vectors(rules, k, total):
            padded = pv + [0] * (m - k)
            # §5.6a: no fee before the first creditor payment date. Payments
            # start at dates[0], so every cadence date is eligible.
            for fees in (fee_splits(fee_total, m) if fee_total else [[0] * m]):
                if simulate(client, dates, padded, fees, rules.bank_fee_cents, extra) is None:
                    continue
                cum, acc = [], 0
                for f in fees:
                    acc += f
                    cum.append(acc)
                cand = (tuple(-c for c in cum), dates, padded, fees)
                if best is None or cand[0] < best[0]:
                    best = cand
                if first_only:
                    return best
    return best


def oracle_feasible(client, offer, rules, extra=()) -> bool:
    return best_schedule(client, offer, rules, extra=extra, first_only=True) is not None


def cumulative_fee(rows, dates) -> list[int]:
    idx = {d: i for i, d in enumerate(dates)}
    fv = [0] * len(dates)
    for r in rows:
        fv[idx[r.date]] = r.program_fee_cents
    out, acc = [], 0
    for f in fv:
        acc += f
        out.append(acc)
    return out


def assert_conforms(result, client, offer, rules):
    """Re-check all ten §5 constraints against a feasible Result, from scratch."""
    dates = cadence(client, offer)
    total = rnd(offer.settlement_pct, offer.creditor_balance_cents)
    fee_total = rnd(rules.program_fee_pct, offer.original_balance_cents)
    rows = result.schedule
    assert rows, "a feasible result must carry a schedule"

    pay_rows = [r for r in rows if r.creditor_payment_cents > 0]
    payments = [r.creditor_payment_cents for r in pay_rows]
    k = len(payments)

    # 1. consecutive cadence dates from first_payment_date; nothing past horizon
    assert 1 <= k <= min(rules.max_payments, rules.max_terms)
    assert [r.date for r in pay_rows] == dates[:k]
    assert all(r.date <= client.last_draft_date for r in rows)
    # 2. exact sum
    assert sum(payments) == total
    # 3. non-decreasing
    assert all(a <= b for a, b in zip(payments, payments[1:]))
    # 4. floors
    assert floors_ok(payments, rules)
    # 5. bank fee on payment dates only
    assert all(r.bank_fee_cents == rules.bank_fee_cents for r in pay_rows)
    assert all(r.bank_fee_cents == 0 for r in rows if r.creditor_payment_cents == 0)
    # 6. fee timing
    assert sum(r.program_fee_cents for r in rows) == fee_total
    assert all(r.program_fee_cents >= 0 for r in rows)
    assert all(r.date >= pay_rows[0].date for r in rows if r.program_fee_cents > 0)
    # 7/8/9. shape, and the shape *reported* in pay_shape_used
    assert shape_ok(payments, rules)
    if rules.even_pays or payments == even_split(sum(payments), len(payments)):
        expected_shape = "even"
    elif is_balloon(payments):
        expected_shape = "balloon"
    else:
        expected_shape = "staircase"
    assert result.pay_shape_used == expected_shape, (
        f"reported {result.pay_shape_used} for {payments}, expected {expected_shape}")
    if result.pay_shape_used == "balloon":
        assert rules.is_ballooning_allowed
    # 10. simulation, credits before debits, balance >= 0 everywhere
    idx = {d: i for i, d in enumerate(dates)}
    pv = [0] * len(dates)
    fv = [0] * len(dates)
    for r in rows:
        pv[idx[r.date]] = r.creditor_payment_cents
        fv[idx[r.date]] = r.program_fee_cents
    balances = simulate(client, dates, pv, fv, rules.bank_fee_cents)
    assert balances is not None, "balance went negative"
    for r in rows:
        assert r.balance_cents == balances[r.date], f"reported balance wrong on {r.date}"


# ---------------------------------------------------------------------------
# Scenario builders for the randomized cross-checks
# ---------------------------------------------------------------------------

def _objective_scenario(rng):
    m = rng.randint(2, 3)
    draft = rng.randint(3, 8)
    ledger = [LedgerEntry(date(2026, 1 + i, 1), draft, "credit") for i in range(m)]
    if rng.random() < 0.3:
        ledger.append(LedgerEntry(date(2026, 1 + rng.randrange(m), 15),
                                  rng.randint(1, 3), "debit"))
    client = Client(draft, 1, date(2026, 1, 1), date(2026, m, 28),
                    date(2025, 12, 31), rng.choice([0, 2]), ledger)
    total = rng.randint(2, 16)
    offer = Offer("Rand", total, 8, 1.0, date(2026, 1, 28))
    rules = CreditorRules(
        max_terms=rng.randint(1, m), max_payments=rng.randint(1, m),
        min_payment_cents=rng.choice([0, 1, 2]),
        max_token_pays=rng.randint(0, 3),
        min_payment_tiers=rng.choice([[], [(2, 3)], [(3, 4)]]),
        even_pays=rng.random() < 0.25,
        is_ballooning_allowed=rng.random() < 0.25,
        max_segments=rng.randint(1, 3),
        bank_fee_cents=rng.choice([0, 1]),
        program_fee_pct=rng.choice([0.0, 0.25, 0.5]),
    )
    return client, offer, rules


def _minima_scenario(rng):
    m = rng.randint(2, 3)
    draft = rng.randint(2, 5)
    ledger = [LedgerEntry(date(2026, 1 + i, 1), draft, "credit") for i in range(m)]
    client = Client(draft, 1, date(2026, 1, 1), date(2026, m, 28),
                    date(2025, 12, 31), 0, ledger)
    total = rng.randint(4, 16)
    offer = Offer("Rand", total, 8, 1.0, date(2026, 1, 28))
    rules = CreditorRules(m, m, rng.choice([0, 2]), rng.randint(0, 3), [],
                          rng.random() < 0.3, rng.random() < 0.3,
                          rng.randint(1, 3), rng.choice([0, 1]),
                          rng.choice([0.0, 0.25]))
    return client, offer, rules


# ---------------------------------------------------------------------------
# Regressions: the fee greedy must not strand a later creditor payment
# ---------------------------------------------------------------------------

def test_fee_greedy_leaves_room_for_a_later_creditor_payment():
    """The fee held at date i is still held later, while payments keep growing.

    cap = [9, 16, 23], payments [7, 8], fee 2. Taking the full fee on date 0
    (room 9-7 = 2) leaves 15+2 = 17 > 16 on date 1. The headroom that binds is
    the suffix minimum of ``cap[i] - cumulative_payments[i]``, not of ``cap``
    alone; fees [1, 0, 1] is feasible and must be found.
    """
    client = Client(7, 1, date(2026, 1, 1), date(2026, 3, 28), date(2025, 12, 31), 2,
                    [LedgerEntry(date(2026, 1 + i, 1), 7, "credit") for i in range(3)])
    offer = Offer("Squeeze", 15, 8, 1.0, date(2026, 1, 28))
    rules = CreditorRules(2, 2, 1, 3, [], False, False, 1, 0, 0.25)

    result = evaluate_offer(client, offer, rules)
    assert result.feasible, "a valid schedule exists; the fee greedy overspent early"
    assert_conforms(result, client, offer, rules)
    assert [r.creditor_payment_cents for r in result.schedule if r.creditor_payment_cents] == [7, 8]
    # fee is still collected as early as the cash genuinely allows
    assert cumulative_fee(result.schedule, cadence(client, offer)) == [1, 1, 2]


def test_lump_minimum_is_not_inflated_by_the_fee_greedy():
    """Same defect seen through Part 2: an over-eager fee grab inflates L.

    With a 2c lump the schedule [1,1,2] with the fee taken last is feasible, so
    the minimum lump is 2 — not 3.
    """
    client = Client(2, 1, date(2026, 1, 1), date(2026, 3, 28), date(2025, 12, 31), 0,
                    [LedgerEntry(date(2026, 1 + i, 1), 2, "credit") for i in range(3)])
    offer = Offer("Tight", 4, 8, 1.0, date(2026, 1, 28))
    rules = CreditorRules(3, 3, 0, 1, [], True, False, 2, 1, 0.25)

    result = evaluate_offer(client, offer, rules)
    assert not result.feasible
    lump = result.additional_funds.lump_sum
    assert lump.amount_cents == 2
    # and it really is minimal, per the independent oracle
    at = [LedgerEntry(lump.date, lump.amount_cents, "credit")]
    below = [LedgerEntry(lump.date, lump.amount_cents - 1, "credit")]
    assert oracle_feasible(client, offer, rules, extra=at)
    assert not oracle_feasible(client, offer, rules, extra=below)


@pytest.mark.parametrize("settlement_pct,creditor_balance,expected_total", [
    (0.5, 100000, 50000),     # exact
    (0.4, 150000, 60000),     # exact
    (0.333, 100000, 33300),   # truncating decimal
    (0.5, 12345, 6173),       # .5 -> rounds AWAY from zero, not to even
    (0.07, 114500, 8015),     # would drift if multiplied as floats
])
def test_payments_sum_exactly_to_the_rounded_offer_total(
        settlement_pct, creditor_balance, expected_total):
    """§5.2 against §3's rounding: the sum must hit round-half-up(pct * balance).

    Deliberately includes totals that do not divide by k and totals whose last
    cent depends on rounding half-up rather than half-to-even.
    """
    ledger = [LedgerEntry(date(2026, m, 1), 40000, "credit") for m in range(1, 13)]
    client = Client(40000, 1, date(2026, 1, 1), date(2026, 12, 1),
                    date(2025, 12, 31), 0, ledger)
    offer = Offer("Sum", creditor_balance, 100000, settlement_pct, date(2026, 1, 31))
    rules = CreditorRules(12, 12, 2500, 6, [], False, False, 3, 500, 0.1)

    assert rnd(settlement_pct, creditor_balance) == expected_total
    result = evaluate_offer(client, offer, rules)
    assert result.feasible
    payments = [r.creditor_payment_cents for r in result.schedule
                if r.creditor_payment_cents > 0]
    assert sum(payments) == expected_total
    assert_conforms(result, client, offer, rules)


@pytest.mark.parametrize("balloon_allowed", [False, True])
def test_an_indivisible_even_schedule_is_reported_even(balloon_allowed):
    """§7: "as equal as possible" IS the even shape.

    The reported shape must not hinge on whether k divides offer_total. With a
    7c offer over two payments the only legal schedule is [3, 4]; that is even
    in exactly the way [3, 3] would be for 6c -- not a staircase, and certainly
    not a balloon absorbing the remaining balance.
    """
    client = Client(6, 1, date(2026, 1, 1), date(2026, 2, 28), date(2025, 12, 31), 0,
                    [LedgerEntry(date(2026, 1, 1), 6, "credit"),
                     LedgerEntry(date(2026, 2, 1), 6, "credit")])
    offer = Offer("Odd", 7, 8, 1.0, date(2026, 1, 28))
    rules = CreditorRules(2, 2, 3, 2, [], False, balloon_allowed, 2, 0, 0.0)

    result = evaluate_offer(client, offer, rules)
    assert result.feasible
    assert [r.creditor_payment_cents for r in result.schedule] == [3, 4]
    assert result.pay_shape_used == "even"
    assert_conforms(result, client, offer, rules)


def test_a_one_cent_remainder_is_not_a_balloon():
    """§2: a balloon absorbs the *entire remaining balance*.

    [2, 2, 3] is even_split(7, 3) -- the +1c lands on the latest payment purely
    because 7 is not divisible by 3. Reporting that as a balloon would be wrong
    even where the creditor permits ballooning.
    """
    rules_balloon = CreditorRules(9, 9, 0, 9, [], False, True, 4, 0, 0.0)
    rules_plain = CreditorRules(9, 9, 0, 9, [], False, False, 4, 0, 0.0)
    from feasibility.solver import classify_shape
    for payments in ([2, 2, 3], [3, 4], [8333, 8334], [5, 5, 5]):
        assert not is_balloon(payments)
        assert classify_shape(rules_balloon, payments) == "even"
        assert classify_shape(rules_plain, payments) == "even"
    # a genuine balloon still reads as one
    assert is_balloon([25, 25, 25, 175])
    assert classify_shape(rules_balloon, [25, 25, 25, 175]) == "balloon"
    # ...and a real step up is a staircase, not a balloon
    assert classify_shape(rules_plain, [25, 75, 75]) == "staircase"


def test_negative_opening_balance_is_reported_as_a_cash_shortfall():
    """A negative opening balance is a *cash* problem, not a rules problem.

    No credit dated after ``as_of_date`` can repair a balance that already went
    negative on it, so the offer is infeasible whatever we fund -- but blaming
    "dates, floors, or segment limits" would point the reader at the wrong thing.
    """
    client = Client(10000, 1, date(2026, 1, 1), date(2026, 6, 1),
                    date(2025, 12, 31), -5000,
                    [LedgerEntry(date(2026, m, 1), 10000, "credit")
                     for m in range(1, 7)])
    offer = Offer("Overdrawn", 30000, 30000, 0.5, date(2026, 1, 31))
    rules = CreditorRules(6, 6, 2500, 6, [], False, False, 4, 0, 0.1)

    result = evaluate_offer(client, offer, rules)
    assert not result.feasible
    af = result.additional_funds
    for option in (af.lump_sum, af.monthly_increment):
        assert option.amount_cents == 0
        assert option.within_guardrail is False
        assert "-5000" in option.reason and "2025-12-31" in option.reason
        assert "floors" not in option.reason


def _degenerate_cases():
    """Inputs the spec never contemplates. None may crash or emit garbage."""
    drafts = [LedgerEntry(date(2026, m, 1), 10000, "credit") for m in range(1, 7)]

    def client(**kw):
        base = dict(draft_amount_cents=10000, draft_day=1,
                    first_draft_date=date(2026, 1, 1), last_draft_date=date(2026, 6, 1),
                    as_of_date=date(2025, 12, 31), current_balance_cents=0,
                    ledger=list(drafts))
        base.update(kw)
        return Client(**base)

    def offer(**kw):
        base = dict(creditor="X", current_balance_cents=30000,
                    original_balance_cents=30000, settlement_pct=0.5,
                    first_payment_date=date(2026, 1, 31))
        base.update(kw)
        return Offer(**base)

    def rules(**kw):
        base = dict(max_terms=6, max_payments=6, min_payment_cents=2500,
                    max_token_pays=6, min_payment_tiers=[], even_pays=False,
                    is_ballooning_allowed=False, max_segments=4,
                    bank_fee_cents=0, program_fee_pct=0.1)
        base.update(kw)
        return CreditorRules(**base)

    return {
        "negative-opening-balance": (client(current_balance_cents=-5000), offer(), rules()),
        "unsorted-ledger": (client(ledger=list(reversed(drafts))), offer(), rules()),
        "duplicate-ledger-dates": (
            client(ledger=drafts + [LedgerEntry(date(2026, 1, 1), 10000, "credit")]),
            offer(), rules()),
        "entry-on-as-of-date": (
            client(ledger=[LedgerEntry(date(2025, 12, 31), 99999, "credit")] + drafts),
            offer(), rules()),
        "first-payment-on-horizon": (client(), offer(first_payment_date=date(2026, 6, 1)), rules()),
        "first-payment-before-as-of": (client(), offer(first_payment_date=date(2025, 11, 30)), rules()),
        "max-terms-zero": (client(), offer(), rules(max_terms=0)),
        "max-segments-zero": (client(), offer(), rules(max_segments=0)),
        "zero-offer-total": (client(), offer(settlement_pct=0.0), rules()),
        "fee-is-whole-balance": (client(), offer(), rules(program_fee_pct=1.0)),
        "tier-from-first-payment": (client(), offer(), rules(min_payment_tiers=[(1, 5000)])),
        "empty-ledger": (client(ledger=[]), offer(), rules()),
        "huge-token-cap": (client(), offer(), rules(max_token_pays=10 ** 6)),
        "zero-min-payment": (client(), offer(), rules(min_payment_cents=0)),
        "leap-day-cadence": (
            client(first_draft_date=date(2024, 1, 31), last_draft_date=date(2024, 6, 30),
                   as_of_date=date(2023, 12, 31),
                   ledger=[LedgerEntry(d, 10000, "credit") for d in
                           (date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31),
                            date(2024, 4, 30), date(2024, 5, 31), date(2024, 6, 30))]),
            offer(first_payment_date=date(2024, 1, 31)), rules()),
    }


@pytest.mark.parametrize("label", sorted(_degenerate_cases()))
def test_degenerate_inputs_stay_well_formed(label):
    """Never crash; a feasible verdict must still be a fully valid schedule."""
    client, offer, rules = _degenerate_cases()[label]
    result = evaluate_offer(client, offer, rules)
    if result.feasible:
        assert_conforms(result, client, offer, rules)
    else:
        af = result.additional_funds
        assert af is not None and result.schedule is None
        for option in (af.lump_sum, af.monthly_increment):
            assert option.amount_cents >= 0
            # an amount that cannot be met must say why, not fail silently
            assert option.within_guardrail or option.reason
    # and it must serialize
    import json
    json.loads(json.dumps(result.to_dict()))


# ---------------------------------------------------------------------------
# §3 conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pct,cents,expected", [
    (0.5, 1, 1),        # 0.5 -> 1, not 0 (half-to-even would give 0)
    (0.5, 3, 2),        # 1.5 -> 2
    (0.5, 5, 3),        # 2.5 -> 3, not 2
    (0.5, 12345, 6173),
    (0.07, 114500, 8015),   # no float drift
    (0.4, 20000, 8000),
    (0.65, 50000, 32500),
])
def test_rounding_is_half_up(pct, cents, expected):
    from feasibility.models import Offer as _O
    assert rnd(pct, cents) == expected
    # the engine's own helper must agree
    from feasibility.money import pct_of_cents
    assert pct_of_cents(pct, cents) == expected


def test_cadence_matches_the_spec_table():
    from feasibility.models import default_first_payment_date, monthly_payment_dates
    client = Client(20000, 1, date(2026, 1, 1), date(2026, 7, 1), date(2025, 12, 31), 0, [])
    # omitted -> end of month of first_draft_date
    assert default_first_payment_date(client) == date(2026, 1, 31)
    # last day of its month -> true EOM cadence
    assert monthly_payment_dates(date(2026, 1, 31), 4) == [
        date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31), date(2026, 4, 30)]
    assert monthly_payment_dates(date(2024, 1, 31), 2)[1] == date(2024, 2, 29)  # leap day
    # mid-month -> same day each month, clamped, and snapping back afterwards
    assert monthly_payment_dates(date(2026, 1, 15), 3) == [
        date(2026, 1, 15), date(2026, 2, 15), date(2026, 3, 15)]
    assert monthly_payment_dates(date(2026, 1, 30), 3) == [
        date(2026, 1, 30), date(2026, 2, 28), date(2026, 3, 30)]


# ---------------------------------------------------------------------------
# The provided cases, and §9's serialized shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", [
    "case1_feasible_even", "case2_infeasible_minima", "case3_balloon", "case4_tiers",
])
def test_provided_cases_satisfy_every_hard_constraint(case):
    client, offer, rules = load_case(f"cases/{case}")
    result = evaluate_offer(client, offer, rules)
    if result.feasible:
        assert_conforms(result, client, offer, rules)
    else:
        assert result.schedule is None
        assert result.pay_shape_used is None
        assert result.additional_funds is not None


def test_serialized_shape_matches_section_9():
    client, offer, rules = load_case("cases/case1_feasible_even")
    d = evaluate_offer(client, offer, rules).to_dict()
    assert set(d) == {"feasible", "pay_shape_used", "schedule", "additional_funds"}
    assert d["additional_funds"] is None
    assert set(d["schedule"][0]) == {
        "date", "creditor_payment_cents", "program_fee_cents",
        "bank_fee_cents", "balance_cents"}

    client, offer, rules = load_case("cases/case2_infeasible_minima")
    d = evaluate_offer(client, offer, rules).to_dict()
    assert d["schedule"] is None and d["pay_shape_used"] is None
    assert set(d["additional_funds"]) == {"lump_sum", "monthly_increment"}
    assert set(d["additional_funds"]["lump_sum"]) == {
        "amount_cents", "date", "within_guardrail", "reason"}
    assert set(d["additional_funds"]["monthly_increment"]) == {
        "amount_cents", "num_drafts", "within_guardrail", "reason"}


def test_additional_funds_key_set_is_stable_across_outcomes():
    """§9 documents `date` on lump_sum and `num_drafts` on monthly_increment.

    Those keys must be present whatever the outcome — a caller reading
    ``additional_funds["lump_sum"]["date"]`` should get ``null`` when no lump
    sum helps, not a KeyError.
    """
    ledger = [LedgerEntry(date(2026, m, 1), 1000, "credit") for m in range(1, 7)]
    scenarios = {
        # ordinary infeasible: a lump and an increment both exist
        "cash-short": (
            Client(1000, 1, date(2026, 1, 1), date(2026, 6, 1),
                   date(2025, 12, 31), 0, ledger),
            Offer("Short", 500000, 500000, 1.0, date(2026, 1, 31)),
            CreditorRules(6, 6, 2500, 6, [], False, False, 2, 0, 0.1),
        ),
        # structurally infeasible: floors can never sum to the offer total
        "structural": (
            Client(1000, 1, date(2026, 1, 1), date(2026, 6, 1),
                   date(2025, 12, 31), 0, ledger),
            Offer("Imp", 100, 100, 1.0, date(2026, 1, 31)),
            CreditorRules(2, 2, 2500, 0, [], False, False, 1, 0, 0.0),
        ),
        # no future drafts left to raise at all
        "no-drafts": (
            Client(1000, 1, date(2026, 1, 1), date(2026, 6, 1),
                   date(2026, 12, 31), 0, []),
            Offer("Imp", 100, 100, 1.0, date(2026, 1, 31)),
            CreditorRules(2, 2, 2500, 0, [], False, False, 1, 0, 0.0),
        ),
    }
    for label, (client, offer, rules) in scenarios.items():
        result = evaluate_offer(client, offer, rules)
        assert not result.feasible, label
        af = result.to_dict()["additional_funds"]
        assert set(af["lump_sum"]) == {
            "amount_cents", "date", "within_guardrail", "reason"}, label
        assert set(af["monthly_increment"]) == {
            "amount_cents", "num_drafts", "within_guardrail", "reason"}, label
        assert isinstance(af["monthly_increment"]["num_drafts"], int), label
        # and the payload survives a JSON round-trip
        import json
        json.loads(json.dumps(af))


def test_guardrails_use_the_thresholds_from_section_8():
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    af = evaluate_offer(client, offer, rules).additional_funds
    lump_cap = rnd(0.65, rnd(offer.settlement_pct, offer.creditor_balance_cents))
    incr_cap = max(10000, rnd(0.40, client.draft_amount_cents))
    assert af.lump_sum.within_guardrail == (af.lump_sum.amount_cents <= lump_cap)
    assert af.monthly_increment.within_guardrail == (
        af.monthly_increment.amount_cents <= incr_cap)


def test_worked_micro_example_from_section_6():
    """$100 lands before each of 3 dates, offer $250, fee $50, no bank fee."""
    client = Client(10000, 1, date(2026, 1, 1), date(2026, 3, 31), date(2025, 12, 31), 0,
                    [LedgerEntry(date(2026, m, 1), 10000, "credit") for m in (1, 2, 3)])
    offer = Offer("Micro", 25000, 25000, 1.0, date(2026, 1, 31))
    rules = CreditorRules(3, 3, 2500, 3, [], False, False, 4, 0, 0.2)
    result = evaluate_offer(client, offer, rules)
    assert result.feasible
    assert_conforms(result, client, offer, rules)
    # the whole $50 fee is collected on the very first date
    assert result.schedule[0].program_fee_cents == 5000


# ---------------------------------------------------------------------------
# Randomized cross-checks against the exhaustive oracle
# ---------------------------------------------------------------------------

def test_engine_matches_exhaustive_search_on_verdict_and_objective():
    """The engine must agree with brute force on feasibility, and be at least
    as good on the fee-front-loading objective (§6)."""
    rng = random.Random(20260809)
    compared = agreed_infeasible = 0
    for _ in range(250):
        client, offer, rules = _objective_scenario(rng)
        ref = best_schedule(client, offer, rules)
        got = evaluate_offer(client, offer, rules)

        if ref is None:
            assert not got.feasible, "engine found a schedule the oracle says is illegal"
            agreed_infeasible += 1
            continue

        assert got.feasible, "oracle found a valid schedule but the engine reported none"
        assert_conforms(got, client, offer, rules)
        dates = cadence(client, offer)
        cum_ref, acc = [], 0
        for f in ref[3]:
            acc += f
            cum_ref.append(acc)
        assert cumulative_fee(got.schedule, dates) >= cum_ref, "fee not front-loaded enough"
        compared += 1

    assert compared > 100 and agreed_infeasible > 50, "scenario mix degenerated"


def test_reported_minima_are_exactly_minimal():
    """For §8, L must work and L-1 must not; likewise X (oracle-verified)."""
    rng = random.Random(11)
    checked = 0
    for _ in range(120):
        client, offer, rules = _minima_scenario(rng)
        result = evaluate_offer(client, offer, rules)
        if result.feasible:
            continue
        af = result.additional_funds

        lump = af.lump_sum
        if lump.amount_cents:
            assert lump.date <= client.last_draft_date
            at = [LedgerEntry(lump.date, lump.amount_cents, "credit")]
            below = [LedgerEntry(lump.date, lump.amount_cents - 1, "credit")]
            assert oracle_feasible(client, offer, rules, extra=at)
            assert not oracle_feasible(client, offer, rules, extra=below)
            checked += 1

        incr = af.monthly_increment
        future = [e for e in client.ledger
                  if e.type == "credit" and e.date > client.as_of_date]
        assert incr.num_drafts == len(future)
        if incr.amount_cents:
            at = [LedgerEntry(e.date, incr.amount_cents, "credit") for e in future]
            below = [LedgerEntry(e.date, incr.amount_cents - 1, "credit") for e in future]
            assert oracle_feasible(client, offer, rules, extra=at)
            assert not oracle_feasible(client, offer, rules, extra=below)
            checked += 1

    assert checked > 40, "too few infeasible scenarios to be meaningful"
