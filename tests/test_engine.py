"""Full test suite for the feasibility engine.

Covers the checklist in ASSIGNMENT.md §10: even / staircase / balloon shapes,
token-pay and tier floors, the max_segments cap, exact-sum, the date-by-date
simulation (same-day ordering and a balance that hits exactly $0), the horizon
limit, fee compliance, and both Part 2 minima.

The centrepiece is ``assert_valid_schedule``: an independent re-implementation
of every hard constraint in §5, which re-simulates the ledger from scratch
rather than trusting anything the engine reported. Every feasible result in
this file goes through it.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from feasibility.engine import Result, evaluate_offer
from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    add_months,
    load_case,
    offer_total_cents,
    program_fee_cents,
)
from feasibility.money import pct_of_cents, round_half_up
from feasibility.solver import (
    _evaluate_vector,
    cadence_dates,
    cash_caps,
    even_split,
    solve,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def make_client(
    *,
    draft_amount_cents: int = 10000,
    draft_day: int = 1,
    first_draft: str = "2026-01-01",
    last_draft: str = "2026-07-01",
    as_of: str = "2025-12-31",
    current_balance_cents: int = 0,
    extra_entries: list[LedgerEntry] | None = None,
) -> Client:
    """A client whose ledger holds one draft credit per month, inclusive."""
    first, last = date.fromisoformat(first_draft), date.fromisoformat(last_draft)
    ledger: list[LedgerEntry] = []
    when, i = first, 0
    while when <= last:
        ledger.append(LedgerEntry(when, draft_amount_cents, "credit"))
        i += 1
        when = add_months(first, i)
    ledger.extend(extra_entries or [])
    return Client(
        draft_amount_cents=draft_amount_cents,
        draft_day=draft_day,
        first_draft_date=first,
        last_draft_date=last,
        as_of_date=date.fromisoformat(as_of),
        current_balance_cents=current_balance_cents,
        ledger=ledger,
    )


def make_offer(
    *,
    creditor_balance_cents: int = 60000,
    original_balance_cents: int | None = None,
    settlement_pct: float = 0.5,
    first_payment_date: str | None = "2026-01-31",
) -> Offer:
    return Offer(
        creditor="TestCo",
        current_balance_cents=creditor_balance_cents,
        original_balance_cents=(
            creditor_balance_cents
            if original_balance_cents is None
            else original_balance_cents
        ),
        settlement_pct=settlement_pct,
        first_payment_date=(
            date.fromisoformat(first_payment_date) if first_payment_date else None
        ),
    )


def make_rules(
    *,
    max_terms: int = 6,
    max_payments: int = 6,
    min_payment_cents: int = 2500,
    max_token_pays: int = 6,
    min_payment_tiers: list[tuple[int, int]] | None = None,
    even_pays: bool = False,
    is_ballooning_allowed: bool = False,
    max_segments: int = 4,
    bank_fee_cents: int = 0,
    program_fee_pct: float = 0.0,
) -> CreditorRules:
    return CreditorRules(
        max_terms=max_terms,
        max_payments=max_payments,
        min_payment_cents=min_payment_cents,
        max_token_pays=max_token_pays,
        min_payment_tiers=min_payment_tiers or [],
        even_pays=even_pays,
        is_ballooning_allowed=is_ballooning_allowed,
        max_segments=max_segments,
        bank_fee_cents=bank_fee_cents,
        program_fee_pct=program_fee_pct,
    )


# ---------------------------------------------------------------------------
# Independent validator
# ---------------------------------------------------------------------------

def expected_floor(rules: CreditorRules, position: int) -> int:
    """Floor at a 1-based position, derived independently of the engine.

    The ``max(1, ...)`` matches the engine's reading that a creditor payment
    must move at least one cent (see README, "Assumptions").
    """
    floor = rules.min_payment_cents
    if position > rules.max_token_pays:
        floor = rules.min_payment_cents + 1
    for from_payment, min_cents in rules.min_payment_tiers:
        if position >= from_payment:
            floor = max(floor, min_cents)
    return max(1, floor)


def segment_count(payments: list[int]) -> int:
    """Number of payment levels used.

    A level is a maximal run of equal payments. The one exception is the
    "as equal as possible" split of the final level: a trailing run sitting
    exactly one cent above the run before it is that level's rounding
    remainder, not a level of its own (see README, "Segments").
    """
    runs: list[list[int]] = []
    for payment in payments:
        if runs and runs[-1][0] == payment:
            runs[-1][1] += 1
        else:
            runs.append([payment, 1])
    count = len(runs)
    if count >= 2 and runs[-1][0] == runs[-2][0] + 1:
        count -= 1
    return count


def assert_valid_schedule(
    result: Result, client: Client, offer: Offer, rules: CreditorRules
) -> list[int]:
    """Re-check every hard constraint from scratch. Returns the payments."""
    assert result.feasible is True
    assert result.schedule is not None and result.schedule
    assert result.additional_funds is None
    assert result.pay_shape_used in {"even", "staircase", "balloon"}

    horizon = client.last_draft_date
    cadence = cadence_dates(client, offer)
    total = offer_total_cents(offer)
    fee_total = program_fee_cents(offer, rules)
    rows = result.schedule

    # --- 1. count & placement -------------------------------------------
    row_dates = [r.date for r in rows]
    assert row_dates == sorted(set(row_dates)), "dates must be strictly increasing"
    assert set(row_dates) <= set(cadence), "every row must sit on a cadence date"
    assert all(d <= horizon for d in row_dates), "nothing may be scheduled past the horizon"

    payment_rows = [r for r in rows if r.creditor_payment_cents > 0]
    payments = [r.creditor_payment_cents for r in payment_rows]
    k = len(payments)
    assert 1 <= k <= min(rules.max_payments, rules.max_terms)
    # consecutive cadence dates with no gaps, starting at the first one
    assert [r.date for r in payment_rows] == cadence[:k]

    # --- 2. exact sum ----------------------------------------------------
    assert sum(payments) == total

    # --- 3. non-decreasing ------------------------------------------------
    assert all(a <= b for a, b in zip(payments, payments[1:]))

    # --- 4. floors --------------------------------------------------------
    for i, amount in enumerate(payments, start=1):
        assert amount >= expected_floor(rules, i), f"payment {i} below its floor"
    token_pays = sum(1 for p in payments if p == rules.min_payment_cents)
    assert token_pays <= rules.max_token_pays

    # --- 5. bank fee ------------------------------------------------------
    for row in rows:
        expected = rules.bank_fee_cents if row.creditor_payment_cents > 0 else 0
        assert row.bank_fee_cents == expected

    # --- 6. program-fee timing --------------------------------------------
    assert sum(r.program_fee_cents for r in rows) == fee_total
    assert all(r.program_fee_cents >= 0 for r in rows)
    first_payment_date = cadence[0]
    assert all(
        r.program_fee_cents == 0 for r in rows if r.date < first_payment_date
    ), "no program fee before the first creditor payment date"

    # --- 7/8/9. shape -----------------------------------------------------
    if rules.even_pays:
        assert result.pay_shape_used == "even"
        assert max(payments) - min(payments) <= 1, "even pays must be as equal as possible"
        assert payments == sorted(payments)
    elif result.pay_shape_used == "balloon":
        assert rules.is_ballooning_allowed, "balloon requires is_ballooning_allowed"
    elif result.pay_shape_used == "staircase" and not rules.is_ballooning_allowed:
        assert segment_count(payments) <= max(1, rules.max_segments)

    # --- 10. date-by-date simulation --------------------------------------
    events: dict[date, list[tuple[str, int]]] = {}
    for entry in client.ledger:
        if entry.date > client.as_of_date:
            events.setdefault(entry.date, []).append((entry.type, entry.amount_cents))
    for row in rows:
        debits = (
            row.creditor_payment_cents + row.program_fee_cents + row.bank_fee_cents
        )
        if debits:
            events.setdefault(row.date, []).append(("debit", debits))

    balance = client.current_balance_cents
    assert balance >= 0
    by_row = {r.date: r for r in rows}
    for when in sorted(events):
        # same-day ordering: all credits, then all debits
        for kind, amount in events[when]:
            if kind == "credit":
                balance += amount
        for kind, amount in events[when]:
            if kind == "debit":
                balance -= amount
        assert balance >= 0, f"balance went negative on {when}"
        if when in by_row:
            assert by_row[when].balance_cents == balance, f"reported balance wrong on {when}"

    return payments


# ---------------------------------------------------------------------------
# Money: round-half-up
# ---------------------------------------------------------------------------

def test_round_half_up_goes_away_from_zero():
    # Python's built-in round() is half-to-even and would give 2 and 1250 here.
    assert round_half_up(2.5) == 3
    assert round_half_up(1.5) == 2
    assert round_half_up(0.5) == 1
    assert round_half_up(-2.5) == -3
    assert pct_of_cents(0.5, 2501) == 1251


def test_offer_total_and_fee_use_half_up():
    offer = make_offer(creditor_balance_cents=2501, settlement_pct=0.5)
    assert offer_total_cents(offer) == 1251
    rules = make_rules(program_fee_pct=0.5)
    assert program_fee_cents(offer, rules) == 1251


def test_percentages_do_not_drift_through_float():
    # 0.07 * 114500 is 8015.000000000001 in binary floating point.
    offer = make_offer(creditor_balance_cents=114500, settlement_pct=0.07)
    assert offer_total_cents(offer) == 8015


def test_creditor_balance_alias():
    offer = make_offer(creditor_balance_cents=60000)
    assert offer.creditor_balance_cents == offer.current_balance_cents == 60000


# ---------------------------------------------------------------------------
# The four provided cases, fully validated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case,shape",
    [
        ("case1_feasible_even", "even"),
        ("case3_balloon", "balloon"),
        ("case4_tiers", "staircase"),
    ],
)
def test_provided_cases_are_valid(case, shape):
    client, offer, rules = load_case(f"cases/{case}")
    result = evaluate_offer(client, offer, rules)
    assert result.pay_shape_used == shape
    assert_valid_schedule(result, client, offer, rules)


def test_case1_front_loads_the_fee_to_a_zero_balance():
    """The objective in action: the first date is drained to exactly $0."""
    client, offer, rules = load_case("cases/case1_feasible_even")
    result = evaluate_offer(client, offer, rules)
    first = result.schedule[0]
    assert first.balance_cents == 0
    # 20000 in, 8333 to the creditor, 1000 bank -> the other 10667 is our fee.
    assert first.creditor_payment_cents == 8333
    assert first.program_fee_cents == 10667
    # fully collected well before the horizon
    assert sum(r.program_fee_cents for r in result.schedule[:3]) == 30000


def test_case4_tier_and_segment_interaction():
    client, offer, rules = load_case("cases/case4_tiers")
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    # six token pays at the base, then the tier floor forces a step up
    assert payments == [2500] * 6 + [7500] * 6
    assert segment_count(payments) == 2 == rules.max_segments


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_even_shape_distributes_remainder_onto_latest_payments():
    client = make_client(draft_amount_cents=30000)
    offer = make_offer(creditor_balance_cents=100001, settlement_pct=1.0)
    rules = make_rules(even_pays=True, max_payments=4, max_terms=4)
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert result.pay_shape_used == "even"
    # 100001 over 4 -> 25000, 25000, 25000, 25001
    assert payments == [25000, 25000, 25000, 25001]


def test_balloon_defers_everything_to_the_final_payment():
    client = make_client(draft_amount_cents=10000)
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=0.5)
    rules = make_rules(is_ballooning_allowed=True, max_payments=6, max_terms=6)
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert result.pay_shape_used == "balloon"
    assert payments == [2500] * 5 + [17500]


def test_staircase_never_ends_in_a_lone_balloon_payment():
    """Same inputs as the balloon test, with ballooning switched off."""
    client = make_client(draft_amount_cents=10000)
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=0.5)
    rules = make_rules(is_ballooning_allowed=False, max_payments=6, max_terms=6)
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert result.pay_shape_used == "staircase"
    # the final level must be shared by at least two payments
    assert payments.count(payments[-1]) >= 2
    assert payments == [2500, 2500, 2500, 2500, 10000, 10000]


def test_ballooning_flag_ignores_the_segment_cap():
    client = make_client(draft_amount_cents=10000)
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=0.5)
    rules = make_rules(
        is_ballooning_allowed=True, max_segments=1, max_payments=6, max_terms=6
    )
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert result.pay_shape_used == "balloon"
    assert segment_count(payments) == 2 > rules.max_segments


# ---------------------------------------------------------------------------
# Floors: token pays and tiers
# ---------------------------------------------------------------------------

def test_token_pay_cap_forces_later_payments_above_the_base():
    client = make_client(draft_amount_cents=20000)
    offer = make_offer(creditor_balance_cents=20000, settlement_pct=1.0)
    rules = make_rules(
        min_payment_cents=2500, max_token_pays=2, max_payments=4, max_terms=4
    )
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert payments.count(2500) == 2, "at most max_token_pays payments at the base"
    assert all(p > 2500 for p in payments[2:])
    assert payments == [2500, 2500, 7500, 7500]


def test_zero_token_pays_puts_every_payment_strictly_above_the_base():
    client = make_client(draft_amount_cents=20000)
    offer = make_offer(creditor_balance_cents=20000, settlement_pct=1.0)
    rules = make_rules(
        min_payment_cents=2500, max_token_pays=0, max_payments=4, max_terms=4
    )
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert all(p > 2500 for p in payments)


def test_tier_floor_applies_from_its_payment_number_onward():
    client = make_client(draft_amount_cents=20000)
    offer = make_offer(creditor_balance_cents=40000, settlement_pct=1.0)
    rules = make_rules(
        min_payment_cents=2500,
        max_token_pays=6,
        min_payment_tiers=[(3, 8000)],
        max_payments=4,
        max_terms=4,
    )
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert payments[0] == payments[1] == 2500
    assert all(p >= 8000 for p in payments[2:])


def test_overlapping_tiers_take_the_strictest():
    rules = make_rules(min_payment_tiers=[(2, 4000), (2, 9000), (4, 5000)])
    client = make_client(draft_amount_cents=40000)
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=1.0)
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert all(p >= 9000 for p in payments[1:]), "the 9000 tier wins at position 2"


def test_infeasible_when_floors_cannot_sum_to_the_offer():
    """Floors exceed the offer total at every k -> no schedule at any funding."""
    client = make_client(draft_amount_cents=100000)
    offer = make_offer(creditor_balance_cents=10000, settlement_pct=1.0)
    rules = make_rules(min_payment_cents=50000, max_payments=1, max_terms=1)
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    funds = result.additional_funds
    assert funds.lump_sum.amount_cents == 0
    assert funds.lump_sum.within_guardrail is False
    assert "non-cash constraint" in funds.lump_sum.reason
    assert funds.monthly_increment.within_guardrail is False


# ---------------------------------------------------------------------------
# max_segments
# ---------------------------------------------------------------------------

def test_single_segment_forces_one_level():
    client = make_client(draft_amount_cents=20000)
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=1.0)
    rules = make_rules(max_segments=1, max_payments=6, max_terms=6)
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert segment_count(payments) == 1
    assert payments == [10000] * 6


def test_segment_cap_binds_against_the_floor_staircase():
    """Two tiers plus a step-up needs three levels; max_segments=2 forbids it."""
    client = make_client(draft_amount_cents=30000, last_draft="2026-07-01")
    offer = make_offer(creditor_balance_cents=90000, settlement_pct=1.0)
    tiers = [(3, 5000), (5, 9000)]
    loose = make_rules(
        min_payment_tiers=tiers, max_segments=3, max_payments=6, max_terms=6
    )
    tight = make_rules(
        min_payment_tiers=tiers, max_segments=2, max_payments=6, max_terms=6
    )

    loose_payments = assert_valid_schedule(
        evaluate_offer(client, offer, loose), client, offer, loose
    )
    tight_payments = assert_valid_schedule(
        evaluate_offer(client, offer, tight), client, offer, tight
    )
    assert segment_count(loose_payments) == 3
    assert segment_count(tight_payments) <= 2
    # the tighter cap can only cost us: money moves earlier, never later
    assert tight_payments[0] >= loose_payments[0]


def test_more_segments_never_produce_a_worse_prefix():
    client = make_client(draft_amount_cents=25000)
    offer = make_offer(creditor_balance_cents=80000, settlement_pct=1.0)
    prefixes = []
    for segments in (1, 2, 3, 4):
        rules = make_rules(max_segments=segments, max_payments=6, max_terms=6)
        payments = assert_valid_schedule(
            evaluate_offer(client, offer, rules), client, offer, rules
        )
        running, out = 0, []
        for p in payments:
            running += p
            out.append(running)
        prefixes.append(out)
    for looser, tighter in zip(prefixes[1:], prefixes):
        assert all(a <= b for a, b in zip(looser, tighter))


# ---------------------------------------------------------------------------
# Dates: cadence, same-day ordering, horizon
# ---------------------------------------------------------------------------

def test_same_day_credit_is_applied_before_the_debit():
    """The only cash on the payment date is that day's draft, and the payment
    consumes all of it. Feasible only if credits land before debits."""
    client = make_client(
        draft_amount_cents=30000,
        draft_day=31,
        first_draft="2026-01-31",
        last_draft="2026-03-31",
        current_balance_cents=0,
    )
    offer = make_offer(creditor_balance_cents=30000, settlement_pct=1.0)
    rules = make_rules(max_payments=1, max_terms=1, min_payment_cents=0)
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)
    row = result.schedule[0]
    assert row.date == date(2026, 1, 31)
    assert row.creditor_payment_cents == 30000
    assert row.balance_cents == 0


def test_balance_may_touch_exactly_zero_but_not_go_below():
    # Horizon Apr 1 admits the Jan 31 / Feb 28 / Mar 31 cadence.
    client = make_client(draft_amount_cents=10000, last_draft="2026-04-01")
    offer = make_offer(creditor_balance_cents=30000, settlement_pct=1.0)
    rules = make_rules(even_pays=True, max_payments=3, max_terms=3)
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)
    # 10000 in and 10000 out on each of the three cadence dates
    assert [r.balance_cents for r in result.schedule] == [0, 0, 0]


def test_one_cent_short_is_infeasible():
    """The same schedule as above, one cent beyond the client's means."""
    client = make_client(draft_amount_cents=10000, last_draft="2026-04-01")
    offer = make_offer(creditor_balance_cents=30001, settlement_pct=1.0)
    rules = make_rules(even_pays=True, max_payments=3, max_terms=3)
    assert evaluate_offer(client, offer, rules).feasible is False


def test_cadence_stops_at_the_horizon():
    client = make_client(draft_amount_cents=50000, last_draft="2026-03-01")
    offer = make_offer(creditor_balance_cents=40000, settlement_pct=1.0)
    rules = make_rules(max_payments=6, max_terms=6, even_pays=True)
    # Cadence is Jan 31 / Feb 28 only: Mar 31 is past the Mar 1 horizon.
    assert cadence_dates(client, offer) == [date(2026, 1, 31), date(2026, 2, 28)]
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert len(payments) == 2
    assert all(r.date <= client.last_draft_date for r in result.schedule)


def test_first_payment_date_past_the_horizon_is_infeasible():
    client = make_client(last_draft="2026-02-01")
    offer = make_offer(first_payment_date="2026-03-31")
    rules = make_rules()
    assert cadence_dates(client, offer) == []
    assert evaluate_offer(client, offer, rules).feasible is False


def test_omitted_first_payment_date_defaults_to_end_of_month():
    client = make_client(draft_amount_cents=40000, last_draft="2026-03-01")
    offer = make_offer(creditor_balance_cents=30000, settlement_pct=1.0,
                       first_payment_date=None)
    rules = make_rules(even_pays=True, max_payments=2, max_terms=2)
    assert cadence_dates(client, offer)[0] == date(2026, 1, 31)
    result = evaluate_offer(client, offer, rules)
    assert result.schedule[0].date == date(2026, 1, 31)
    assert_valid_schedule(result, client, offer, rules)


def test_end_of_month_cadence_tracks_february():
    client = make_client(draft_amount_cents=40000, last_draft="2026-04-01")
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=1.0,
                       first_payment_date="2026-01-31")
    assert cadence_dates(client, offer) == [
        date(2026, 1, 31),
        date(2026, 2, 28),
        date(2026, 3, 31),
    ]


def test_mid_month_cadence_preserves_the_day():
    client = make_client(
        draft_day=10, first_draft="2026-01-10", last_draft="2026-04-10",
        draft_amount_cents=40000,
    )
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=1.0,
                       first_payment_date="2026-01-15")
    assert cadence_dates(client, offer) == [
        date(2026, 1, 15),
        date(2026, 2, 15),
        date(2026, 3, 15),
    ]


def test_fee_is_held_back_for_a_debit_between_cadence_dates():
    """The balance must stay >= 0 on dates we do not touch.

    A committed debit lands on Feb 1, between the Jan 31 and Feb 28 cadence
    dates, and is larger than the draft that arrives with it. Front-loading the
    fee to the limit of the Jan 31 balance would overdraw the account the next
    day, so the greedy has to leave exactly enough behind.
    """
    client = make_client(
        draft_amount_cents=20000,
        last_draft="2026-05-01",
        extra_entries=[LedgerEntry(date(2026, 2, 1), 25000, "debit")],
    )
    offer = make_offer(
        creditor_balance_cents=20000, original_balance_cents=60000,
        settlement_pct=0.5,
    )
    rules = make_rules(
        program_fee_pct=0.5, max_payments=1, max_terms=1, even_pays=True
    )
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)

    first = result.schedule[0]
    assert first.date == date(2026, 1, 31)
    assert first.creditor_payment_cents == 10000
    # 20000 on hand, 10000 to the creditor: naively the other 10000 is free fee,
    # but 5000 has to survive until Feb 1.
    assert first.program_fee_cents == 5000
    assert first.balance_cents == 5000
    # the whole 30000 fee still lands before the horizon
    assert sum(r.program_fee_cents for r in result.schedule) == 30000
    # Feb 28 carries nothing and is omitted; the fee resumes once cash recovers
    assert [r.date for r in result.schedule] == [
        date(2026, 1, 31),
        date(2026, 3, 31),
        date(2026, 4, 30),
    ]


def test_payment_count_is_chosen_by_the_objective_not_maximised():
    """A large bank fee makes extra payments unaffordable.

    ``max_payments`` is 6 and five cadence dates exist, but each payment costs a
    300.00 bank fee, so the solver settles on two payments. This is the
    trade-off that rules out hard-coding "use the largest k".
    """
    client = make_client(
        draft_amount_cents=1000,
        last_draft="2026-06-01",
        current_balance_cents=100000,
    )
    offer = make_offer(creditor_balance_cents=20000, settlement_pct=1.0)
    rules = make_rules(
        max_payments=6, max_terms=6, bank_fee_cents=30000, min_payment_cents=2500
    )
    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)
    assert len(cadence_dates(client, offer)) == 5
    assert payments == [10000, 10000], "three payments would cost 90000 in bank fees"


def test_preserved_day_cadence_clamps_to_month_length():
    """§3: a mid-month cadence keeps its day-of-month, clamped to the month."""
    client = make_client(
        draft_day=30, first_draft="2026-01-30", last_draft="2026-04-30",
        draft_amount_cents=40000,
    )
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=1.0,
                       first_payment_date="2026-01-30")
    # Jan 30 is not the last day of January, so the day is preserved and only
    # February clamps -- it does not become a true end-of-month cadence.
    assert cadence_dates(client, offer) == [
        date(2026, 1, 30),
        date(2026, 2, 28),
        date(2026, 3, 30),
        date(2026, 4, 30),
    ]


def test_leap_day_cadence():
    client = make_client(
        draft_day=29, first_draft="2028-01-29", last_draft="2028-04-29",
        draft_amount_cents=40000,
    )
    offer = make_offer(creditor_balance_cents=60000, settlement_pct=1.0,
                       first_payment_date="2028-01-29")
    assert cadence_dates(client, offer)[1] == date(2028, 2, 29)


def test_committed_debits_are_respected_not_modified():
    """A fixed debit from another settlement eats the cash we would have used."""
    without = make_client(draft_amount_cents=10000, last_draft="2026-04-01")
    with_debit = make_client(
        draft_amount_cents=10000,
        last_draft="2026-04-01",
        extra_entries=[LedgerEntry(date(2026, 2, 1), 15000, "debit")],
    )
    offer = make_offer(creditor_balance_cents=30000, settlement_pct=1.0)
    rules = make_rules(even_pays=True, max_payments=3, max_terms=3)
    assert evaluate_offer(without, offer, rules).feasible is True
    assert evaluate_offer(with_debit, offer, rules).feasible is False


def test_entries_on_or_before_as_of_date_are_not_double_counted():
    """A past draft is already inside current_balance_cents."""
    client = make_client(
        draft_amount_cents=10000,
        first_draft="2026-01-01",
        last_draft="2026-03-01",
        as_of="2026-01-01",
        current_balance_cents=10000,
    )
    offer = make_offer(creditor_balance_cents=20000, settlement_pct=1.0)
    rules = make_rules(even_pays=True, max_payments=2, max_terms=2)
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)
    # 10000 on hand (the Jan 1 draft, already banked) + 10000 on Feb 1 is
    # exactly the 20000 offer. Counting the Jan 1 draft twice would leave room
    # for more, so one cent over must be infeasible.
    assert evaluate_offer(
        client, make_offer(creditor_balance_cents=20001, settlement_pct=1.0), rules
    ).feasible is False


# ---------------------------------------------------------------------------
# Program fee compliance
# ---------------------------------------------------------------------------

def test_no_fee_before_the_first_creditor_payment():
    client = make_client(draft_amount_cents=20000)
    offer = make_offer(creditor_balance_cents=30000, settlement_pct=1.0)
    rules = make_rules(
        program_fee_pct=0.2, max_payments=3, max_terms=3, even_pays=True
    )
    # Push the first payment out to March: two months of drafts are sitting in
    # the account, but no fee may be collected before the first payment date.
    offer.first_payment_date = date(2026, 3, 31)
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)
    assert result.schedule[0].date == date(2026, 3, 31)
    assert all(r.date >= date(2026, 3, 31) for r in result.schedule)


def test_fee_only_dates_carry_no_bank_fee():
    """The fee spills past the last creditor payment onto fee-only dates."""
    client = make_client(draft_amount_cents=20000, last_draft="2026-06-01")
    offer = make_offer(
        creditor_balance_cents=20000, original_balance_cents=100000,
        settlement_pct=1.0,
    )
    rules = make_rules(
        program_fee_pct=0.25, bank_fee_cents=500, max_payments=2, max_terms=2,
        even_pays=True,
    )
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)
    fee_only = [r for r in result.schedule if r.creditor_payment_cents == 0]
    assert fee_only, "expected at least one fee-only date"
    assert all(r.bank_fee_cents == 0 for r in fee_only)
    assert all(r.program_fee_cents > 0 for r in fee_only)


def test_fee_that_cannot_be_collected_by_the_horizon_is_infeasible():
    client = make_client(draft_amount_cents=10000, last_draft="2026-03-01")
    offer = make_offer(
        creditor_balance_cents=20000, original_balance_cents=100000,
        settlement_pct=1.0,
    )
    # 20000 of settlement + 30000 of fee against 30000 of drafts.
    rules = make_rules(
        program_fee_pct=0.3, max_payments=3, max_terms=3, even_pays=True
    )
    assert evaluate_offer(client, offer, rules).feasible is False


def test_bank_fee_charged_once_per_payment_date():
    client, offer, rules = load_case("cases/case4_tiers")
    result = evaluate_offer(client, offer, rules)
    charged = [r for r in result.schedule if r.bank_fee_cents > 0]
    assert len(charged) == 12
    assert all(r.bank_fee_cents == rules.bank_fee_cents for r in charged)


def test_zero_program_fee_still_produces_a_schedule():
    client, offer, rules = load_case("cases/case3_balloon")
    assert program_fee_cents(offer, rules) == 0
    result = evaluate_offer(client, offer, rules)
    assert_valid_schedule(result, client, offer, rules)
    assert all(r.program_fee_cents == 0 for r in result.schedule)


# ---------------------------------------------------------------------------
# Part 2: minimum additional funds
# ---------------------------------------------------------------------------

def test_case2_minima_match_and_are_minimal():
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    assert result.schedule is None
    funds = result.additional_funds

    assert funds.lump_sum.amount_cents == 10000
    assert funds.lump_sum.date == date(2026, 1, 1)
    assert funds.lump_sum.within_guardrail is True
    assert funds.lump_sum.reason == ""

    assert funds.monthly_increment.amount_cents == 2500
    assert funds.monthly_increment.num_drafts == 5
    assert funds.monthly_increment.within_guardrail is True

    # minimality: one cent less must still be infeasible, and the reported
    # amount must actually work.
    assert _feasible_with_lump(client, offer, rules, 10000) is True
    assert _feasible_with_lump(client, offer, rules, 9999) is False
    assert _feasible_with_increment(client, offer, rules, 2500) is True
    assert _feasible_with_increment(client, offer, rules, 2499) is False


def _feasible_with_lump(client, offer, rules, amount) -> bool:
    when = client.as_of_date + timedelta(days=1)
    entry = LedgerEntry(when, amount, "credit")
    return solve(client, offer, rules, extra=[entry]) is not None


def _feasible_with_increment(client, offer, rules, amount) -> bool:
    extra = [
        LedgerEntry(e.date, amount, "credit")
        for e in client.ledger
        if e.type == "credit" and e.date > client.as_of_date
    ]
    return solve(client, offer, rules, extra=extra) is not None


def test_lump_and_increment_may_imply_different_totals():
    """An increment near the horizon adds cash that arrives too late."""
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    funds = evaluate_offer(client, offer, rules).additional_funds
    lump_total = funds.lump_sum.amount_cents
    increment_total = (
        funds.monthly_increment.amount_cents * funds.monthly_increment.num_drafts
    )
    assert increment_total == 12500 > lump_total == 10000


def test_guardrails_reject_oversized_funding():
    client = make_client(draft_amount_cents=1000, last_draft="2026-03-01")
    offer = make_offer(
        creditor_balance_cents=20000, original_balance_cents=100000,
        settlement_pct=0.5,
    )
    rules = make_rules(
        program_fee_pct=1.0, max_payments=3, max_terms=3, even_pays=True,
        min_payment_cents=0,
    )
    result = evaluate_offer(client, offer, rules)
    assert result.feasible is False
    funds = result.additional_funds

    # lump cap is 65% of the 10000 offer total = 6500
    assert funds.lump_sum.amount_cents > 6500
    assert funds.lump_sum.within_guardrail is False
    assert "exceeds the cap of 6500" in funds.lump_sum.reason

    # increment cap is max(10000, 40% of the 1000 draft) = 10000
    assert funds.monthly_increment.amount_cents > 10000
    assert funds.monthly_increment.within_guardrail is False
    assert "exceeds the cap of 10000" in funds.monthly_increment.reason


def test_guardrail_accepts_an_amount_exactly_at_the_cap():
    """The guardrails reject only what is strictly over the cap."""
    # Three cadence dates fed by 10000/month: 30000 is available by Mar 31.
    client = make_client(draft_amount_cents=10000, last_draft="2026-04-01")
    rules = make_rules(even_pays=True, max_payments=3, max_terms=3)

    offer = make_offer(creditor_balance_cents=36500, settlement_pct=1.0)
    funds = evaluate_offer(client, offer, rules).additional_funds
    assert funds.lump_sum.amount_cents == 6500
    assert pct_of_cents(0.65, 36500) == 23725
    assert funds.lump_sum.within_guardrail is True

    # An offer engineered so the shortfall lands exactly on the lump cap:
    # 85714 - 30000 == round_half_up(0.65 * 85714) == 55714.
    tight_offer = make_offer(creditor_balance_cents=85714, settlement_pct=1.0)
    tight_funds = evaluate_offer(client, tight_offer, rules).additional_funds
    assert pct_of_cents(0.65, 85714) == 55714
    assert tight_funds.lump_sum.amount_cents == 55714
    assert tight_funds.lump_sum.within_guardrail is True, "at the cap, not over it"


def test_lump_is_placed_at_the_earliest_modifiable_date():
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    funds = evaluate_offer(client, offer, rules).additional_funds
    assert funds.lump_sum.date == client.as_of_date + timedelta(days=1)
    assert funds.lump_sum.date <= client.last_draft_date


def test_increment_counts_every_future_draft():
    client = make_client(
        draft_amount_cents=10000,
        first_draft="2026-01-01",
        last_draft="2026-05-01",
        as_of="2026-02-15",
        current_balance_cents=20000,
    )
    offer = make_offer(creditor_balance_cents=200000, settlement_pct=1.0)
    rules = make_rules(max_payments=6, max_terms=6)
    funds = evaluate_offer(client, offer, rules).additional_funds
    # Jan and Feb drafts are already banked; Mar, Apr and May remain.
    assert funds.monthly_increment.num_drafts == 3


def test_infeasible_result_serializes_to_the_documented_shape():
    client, offer, rules = load_case("cases/case2_infeasible_minima")
    payload = evaluate_offer(client, offer, rules).to_dict()
    assert payload["feasible"] is False
    assert payload["pay_shape_used"] is None
    assert payload["schedule"] is None
    assert payload["additional_funds"]["lump_sum"] == {
        "amount_cents": 10000,
        "within_guardrail": True,
        "reason": "",
        "date": "2026-01-01",
    }
    assert payload["additional_funds"]["monthly_increment"] == {
        "amount_cents": 2500,
        "within_guardrail": True,
        "reason": "",
        "num_drafts": 5,
    }


def test_feasible_result_serializes_to_the_documented_shape():
    client, offer, rules = load_case("cases/case1_feasible_even")
    payload = evaluate_offer(client, offer, rules).to_dict()
    assert payload["feasible"] is True
    assert payload["pay_shape_used"] == "even"
    assert payload["additional_funds"] is None
    row = payload["schedule"][0]
    assert set(row) == {
        "date",
        "creditor_payment_cents",
        "program_fee_cents",
        "bank_fee_cents",
        "balance_cents",
    }
    assert row["date"] == "2026-01-31"


# ---------------------------------------------------------------------------
# The assignment's own worked example (§6)
# ---------------------------------------------------------------------------

def test_worked_micro_example_from_the_assignment():
    """§6: three cadence dates, $100 lands before each, start $0.

    offer_total $250, program_fee $50, bank_fee $0, flat min $25. The
    assignment notes that ``[$50, $100, $100]`` is valid and collects the whole
    fee on the first date.

    Putting the cadence on the draft day is what reproduces "money lands before
    each date": credits precede debits, and the horizon is the last draft.
    """
    client = make_client(
        draft_amount_cents=10000,
        first_draft="2026-01-01",
        last_draft="2026-03-01",
    )
    offer = make_offer(
        creditor_balance_cents=50000,
        original_balance_cents=25000,
        settlement_pct=0.5,
        first_payment_date="2026-01-01",
    )
    rules = make_rules(
        min_payment_cents=2500, max_payments=3, max_terms=3, program_fee_pct=0.2
    )
    assert offer_total_cents(offer) == 25000
    assert program_fee_cents(offer, rules) == 5000
    assert cadence_dates(client, offer) == [
        date(2026, 1, 1),
        date(2026, 2, 1),
        date(2026, 3, 1),
    ]

    result = evaluate_offer(client, offer, rules)
    payments = assert_valid_schedule(result, client, offer, rules)

    # The point of the example: the entire fee is collected on the first date.
    assert result.schedule[0].program_fee_cents == 5000

    # We beat the assignment's illustrative [5000, 10000, 10000]: it ties on the
    # objective (that schedule also collects the full fee on date one), so the
    # tie-break decides, and §6 asks us to "keep creditor payments as low as the
    # rules allow early on".
    assert payments == [2500, 11250, 11250]
    assert payments[0] < 5000
    assert sum(payments) == 25000


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

def final_level_size(payments: list[int]) -> int:
    """How many payments share the last level (counting its +1-cent tail)."""
    runs: list[list[int]] = []
    for payment in payments:
        if runs and runs[-1][0] == payment:
            runs[-1][1] += 1
        else:
            runs.append([payment, 1])
    size = runs[-1][1]
    if len(runs) >= 2 and runs[-1][0] == runs[-2][0] + 1:
        size += runs[-2][1]
    return size


def _all_non_decreasing(total: int, floors: list[int]):
    """Every non-decreasing vector at or above ``floors`` summing to ``total``."""
    k = len(floors)

    def walk(i: int, previous: int, left: int):
        if i == k:
            if left == 0:
                yield []
            return
        low = max(previous, floors[i])
        # every remaining position must also be at least this value
        for value in range(low, left // (k - i) + 1):
            for rest in walk(i + 1, value, left - value):
                yield [value] + rest

    yield from walk(0, 0, total)


def _shape_allowed(rules: CreditorRules, payments: list[int], total: int) -> bool:
    """The shape rules of §5.7-5.9, as interpreted in the README."""
    k = len(payments)
    if rules.even_pays:
        return payments == even_split(total, k)
    if rules.is_ballooning_allowed:
        return True  # max_segments is ignored, any final payment may absorb
    if segment_count(payments) > max(1, rules.max_segments):
        return False
    if k >= 2 and final_level_size(payments) < 2:
        return False  # that would be a balloon, which this creditor forbids
    return True


def _brute_force_plan(client: Client, offer: Offer, rules: CreditorRules):
    """The optimum by exhaustive search over every legal payment vector."""
    cadence = cadence_dates(client, offer)
    if not cadence:
        return None
    caps = cash_caps(client, cadence)
    if not caps.ok_before_first:
        return None
    total = offer_total_cents(offer)
    fee_total = program_fee_cents(offer, rules)

    best = None
    k_max = min(rules.max_payments, rules.max_terms, len(cadence))
    for k in range(1, k_max + 1):
        floors = [expected_floor(rules, i) for i in range(1, k + 1)]
        for payments in _all_non_decreasing(total, floors):
            if not _shape_allowed(rules, payments, total):
                continue
            plan = _evaluate_vector(cadence, caps, rules, payments, fee_total)
            if plan is not None and (best is None or plan.key < best.key):
                best = plan
    return best


# Tiny amounts keep the exhaustive search cheap; cents are just integers, so
# nothing about the logic depends on their scale.
BRUTE_FORCE_RULES = [
    make_rules(min_payment_cents=1, max_payments=4, max_terms=4),
    make_rules(min_payment_cents=1, max_payments=4, max_terms=4, max_segments=1),
    make_rules(min_payment_cents=1, max_payments=4, max_terms=4, max_segments=2),
    make_rules(min_payment_cents=1, max_payments=4, max_terms=4, max_segments=3),
    make_rules(min_payment_cents=1, max_payments=4, max_terms=4, even_pays=True),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4, is_ballooning_allowed=True
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4, max_token_pays=1
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4, max_token_pays=2,
        max_segments=2,
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4,
        min_payment_tiers=[(3, 4)],
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4,
        min_payment_tiers=[(2, 3), (4, 5)], max_segments=2,
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4, bank_fee_cents=1,
        program_fee_pct=0.25,
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4, bank_fee_cents=1,
        program_fee_pct=0.5, max_segments=2,
    ),
    make_rules(
        min_payment_cents=1, max_payments=4, max_terms=4,
        is_ballooning_allowed=True, program_fee_pct=0.25, bank_fee_cents=1,
    ),
]


@pytest.mark.parametrize("rules", BRUTE_FORCE_RULES, ids=range(len(BRUTE_FORCE_RULES)))
@pytest.mark.parametrize("total", [4, 7, 12, 13, 20])
@pytest.mark.parametrize("draft", [3, 5, 8])
def test_search_matches_exhaustive_enumeration(rules, total, draft):
    """The pruned shape search finds the same optimum as brute force.

    This is the real test of the candidate generation: the block-structure
    reductions in ``staircase_candidates`` and the claim that the balloon
    dominates every other vector when ballooning is allowed.
    """
    client = make_client(draft_amount_cents=draft, last_draft="2026-05-01")
    offer = make_offer(
        creditor_balance_cents=total, original_balance_cents=8, settlement_pct=1.0
    )
    engine_plan = solve(client, offer, rules)
    brute_plan = _brute_force_plan(client, offer, rules)

    assert (engine_plan is None) == (brute_plan is None), "feasibility disagreement"
    if engine_plan is None:
        return
    assert engine_plan.key == brute_plan.key, (
        f"engine chose {engine_plan.payments} but "
        f"{brute_plan.payments} scores better"
    )


def test_fuzz_every_feasible_result_is_valid():
    """Randomised end-to-end check against the independent validator."""
    rng = random.Random(20260807)
    feasible_seen = infeasible_seen = 0

    for _ in range(300):
        months = rng.randint(2, 8)
        draft = rng.randrange(1000, 30000, 500)
        client = make_client(
            draft_amount_cents=draft,
            last_draft=add_months(date(2026, 1, 1), months - 1).isoformat(),
            current_balance_cents=rng.choice([0, 0, 5000, 20000]),
            extra_entries=(
                [
                    LedgerEntry(
                        add_months(date(2026, 1, 1), rng.randrange(months)),
                        rng.randrange(1000, 20000, 500),
                        "debit",
                    )
                ]
                if rng.random() < 0.35
                else []
            ),
        )
        max_pays = rng.randint(1, 8)
        rules = make_rules(
            max_terms=max_pays,
            max_payments=max_pays,
            min_payment_cents=rng.choice([0, 1000, 2500, 5000]),
            max_token_pays=rng.randint(0, 6),
            min_payment_tiers=rng.choice([[], [(3, 5000)], [(2, 4000), (5, 9000)]]),
            even_pays=rng.random() < 0.25,
            is_ballooning_allowed=rng.random() < 0.25,
            max_segments=rng.randint(1, 4),
            bank_fee_cents=rng.choice([0, 500, 1000]),
            program_fee_pct=rng.choice([0.0, 0.1, 0.25]),
        )
        offer = make_offer(
            creditor_balance_cents=rng.randrange(10000, 150000, 1000),
            original_balance_cents=rng.randrange(10000, 150000, 1000),
            settlement_pct=rng.choice([0.25, 0.4, 0.5, 0.65]),
            first_payment_date=rng.choice(["2026-01-31", "2026-01-15", None]),
        )

        result = evaluate_offer(client, offer, rules)
        if result.feasible:
            feasible_seen += 1
            assert_valid_schedule(result, client, offer, rules)
        else:
            infeasible_seen += 1
            assert result.schedule is None
            assert result.additional_funds is not None

    # the corpus must actually exercise both branches
    assert feasible_seen > 30
    assert infeasible_seen > 30


def test_fuzz_reported_minima_are_exactly_minimal():
    """For infeasible cases, L works and L-1 does not (likewise X)."""
    rng = random.Random(4242)
    checked = 0

    for _ in range(120):
        months = rng.randint(2, 6)
        client = make_client(
            draft_amount_cents=rng.randrange(1000, 20000, 500),
            last_draft=add_months(date(2026, 1, 1), months - 1).isoformat(),
        )
        max_pays = rng.randint(1, 6)
        rules = make_rules(
            max_terms=max_pays,
            max_payments=max_pays,
            min_payment_cents=rng.choice([0, 2500]),
            max_token_pays=rng.randint(0, 6),
            even_pays=rng.random() < 0.3,
            is_ballooning_allowed=rng.random() < 0.3,
            max_segments=rng.randint(1, 4),
            bank_fee_cents=rng.choice([0, 500]),
            program_fee_pct=rng.choice([0.0, 0.2]),
        )
        offer = make_offer(
            creditor_balance_cents=rng.randrange(20000, 200000, 1000),
            original_balance_cents=rng.randrange(20000, 200000, 1000),
            settlement_pct=rng.choice([0.4, 0.5]),
        )

        result = evaluate_offer(client, offer, rules)
        if result.feasible:
            continue
        funds = result.additional_funds

        lump = funds.lump_sum
        if lump.date is not None and lump.amount_cents > 0:
            checked += 1
            assert _feasible_with_lump(client, offer, rules, lump.amount_cents)
            assert not _feasible_with_lump(
                client, offer, rules, lump.amount_cents - 1
            )

        increment = funds.monthly_increment
        if increment.amount_cents > 0:
            assert _feasible_with_increment(
                client, offer, rules, increment.amount_cents
            )
            assert not _feasible_with_increment(
                client, offer, rules, increment.amount_cents - 1
            )

    assert checked > 20
