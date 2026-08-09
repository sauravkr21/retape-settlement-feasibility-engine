"""Solver internals: cadence, cash caps, payment shapes, fee placement, minima.

The public entry point is ``feasibility.engine.evaluate_offer``; this module
holds the modelling. See the "Implementation notes" section of README.md for
the reasoning behind the objective and the shape interpretations.

Model in one paragraph
----------------------
Creditor payments occupy the first ``k`` cadence dates; the program fee may sit
on any cadence date from the first payment date through the horizon. Because
our debits are the only movable entries, the whole feasibility question reduces
to a prefix constraint: at cadence date ``i``, everything we have taken out so
far must not exceed the cash the account has seen by then. Minimising those
prefix totals is exactly what "collect the fee as early as possible" asks for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations

from feasibility.models import (
    Client,
    CreditorRules,
    LedgerEntry,
    Offer,
    default_first_payment_date,
    monthly_payment_dates,
    offer_total_cents,
    program_fee_cents,
)

# Safety rail: a cadence can never be longer than this many months.
MAX_CADENCE_DATES = 1200
# Safety rail for the additional-funds binary searches (cents).
MAX_FUNDING_SEARCH = 10**13


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------

def cadence_dates(client: Client, offer: Offer) -> list[date]:
    """Every cadence date at or before the horizon (``last_draft_date``).

    Creditor payments take a consecutive prefix of these; program fees may land
    on any of them.
    """
    start = offer.first_payment_date or default_first_payment_date(client)
    horizon = client.last_draft_date
    if start > horizon:
        return []
    count = 1
    while count < MAX_CADENCE_DATES:
        if monthly_payment_dates(start, count + 1)[-1] > horizon:
            break
        count += 1
    return monthly_payment_dates(start, count)


# ---------------------------------------------------------------------------
# Cash availability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CashCaps:
    """Cash headroom for our debits, indexed by cadence position.

    ``balance_at[i]`` is the committed running balance on cadence date ``i``
    (credits applied, our debits not yet). ``cap[i]`` is the smallest committed
    balance from cadence date ``i`` up to just before date ``i+1`` — our
    cumulative debits through ``i`` must not exceed it, or the account dips
    negative somewhere in that window.

    Taking the window minimum rather than the balance on ``d_i`` is what
    enforces "balance >= 0 at *every* date" and not merely on the dates we
    touch: a committed debit landing between two cadence dates is caught here.
    """

    ok_before_first: bool
    balance_at: list[int]
    cap: list[int]


def cash_caps(
    client: Client,
    cadence: list[date],
    extra: tuple[LedgerEntry, ...] | list[LedgerEntry] = (),
) -> CashCaps:
    """Fold the committed ledger (plus any injected funding) into cash caps."""
    net: dict[date, int] = {}
    for entry in list(client.ledger) + list(extra):
        # Entries on or before as_of_date are already inside current_balance_cents.
        if entry.date <= client.as_of_date:
            continue
        delta = entry.amount_cents if entry.type == "credit" else -entry.amount_cents
        net[entry.date] = net.get(entry.date, 0) + delta

    checkpoints = sorted(set(net) | set(cadence))
    running = client.current_balance_cents
    balance: dict[date, int] = {}
    for d in checkpoints:
        # Credits before debits on a date is automatic here: the two are netted,
        # and a date is only non-negative if it is non-negative after all of its
        # debits, which is the same test either way.
        running += net.get(d, 0)
        balance[d] = running

    first = cadence[0]
    ok_before_first = client.current_balance_cents >= 0 and all(
        balance[d] >= 0 for d in checkpoints if d < first
    )

    cap: list[int] = []
    for i, d in enumerate(cadence):
        upper = cadence[i + 1] if i + 1 < len(cadence) else None
        window = [
            balance[t]
            for t in checkpoints
            if t >= d and (upper is None or t < upper)
        ]
        cap.append(min(window))

    return CashCaps(
        ok_before_first=ok_before_first,
        balance_at=[balance[d] for d in cadence],
        cap=cap,
    )


# ---------------------------------------------------------------------------
# Payment floors and shapes
# ---------------------------------------------------------------------------

def payment_floors(rules: CreditorRules, k: int) -> list[int]:
    """The per-position minimum for a ``k``-payment schedule.

    Combines the three floor sources of constraint 4. The token-pay rule turns
    into a per-position floor because payments are non-decreasing: every payment
    equal to the base minimum therefore forms a prefix, so a payment past
    position ``max_token_pays`` cannot be at the base and must strictly exceed
    it (i.e. ``base + 1`` cent).

    The result is cumulative-maximum'd: a non-decreasing sequence respecting
    floor ``F_i`` at each position also respects every earlier floor.

    Every payment is also at least one cent. A zero-cent "payment" would let a
    schedule quietly start after ``first_payment_date`` while pretending to
    occupy it, which constraint 1 does not allow (and it would draw a bank fee
    for moving no money).
    """
    base = rules.min_payment_cents
    floors: list[int] = []
    for i in range(1, k + 1):
        floor = base + 1 if i > rules.max_token_pays else base
        for from_payment, min_cents in rules.min_payment_tiers:
            if i >= from_payment:
                floor = max(floor, min_cents)
        floors.append(max(1, floor))

    out: list[int] = []
    high = 0
    for floor in floors:
        high = max(high, floor)
        out.append(high)
    return out


def even_split(total: int, n: int) -> list[int]:
    """"As equal as possible", remainder cents on the latest payments."""
    quotient, remainder = divmod(total, n)
    return [quotient] * (n - remainder) + [quotient + 1] * remainder


def _blocks_to_payments(
    ends: tuple[int, ...], k: int, total: int, floors: list[int]
) -> list[int] | None:
    """Build the payment vector for one block structure.

    ``ends`` are the 1-based last positions of the non-final blocks; the final
    block runs from ``ends[-1] + 1`` to ``k``. A non-final block sits at the
    highest floor it covers (the cheapest value it may legally take), and the
    final block absorbs whatever is left, split as equally as possible.
    """
    payments: list[int] = []
    prev = 0
    for end in ends:
        payments.extend([floors[end - 1]] * (end - prev))
        prev = end

    length = k - prev
    remaining = total - sum(payments)
    if remaining < 0:
        return None
    tail = even_split(remaining, length)
    # Check the floors position by position rather than against the highest
    # floor in the block: when the block spans a floor step-up, the remainder
    # cents of the "as equal as possible" split may land exactly on the raised
    # positions, which is legal.
    for offset, value in enumerate(tail):
        if value < floors[prev + offset]:
            return None
    if payments and tail[0] < payments[-1]:
        return None
    payments.extend(tail)
    return payments


def staircase_candidates(
    rules: CreditorRules, k: int, total: int, floors: list[int]
) -> list[list[int]]:
    """Every staircase worth considering for this ``k``.

    A staircase is a partition of the payments into at most ``max_segments``
    consecutive blocks. Two reductions keep the enumeration tiny without losing
    the optimum:

    1. A non-final block's value is pinned to the highest floor it covers —
       raising it only shifts money earlier, which the objective never wants.
    2. A non-final block that ends mid-plateau can be extended to the end of
       that plateau at no cost, and doing so hands the positions it swallows a
       *lower* value than the following block would have given them. So block
       ends only need to be considered at floor plateau ends — plus ``k-2``,
       the latest start the final block may have.
    """
    segments = max(1, rules.max_segments)
    if k == 1:
        return [[total]] if total >= floors[0] else []

    # 1-based positions where the floor steps up on the next payment.
    plateau_ends = [i for i in range(1, k) if floors[i] != floors[i - 1]]
    # A final block of length 1 would be a balloon, which needs the flag.
    candidates = sorted({*plateau_ends, k - 2})
    candidates = [e for e in candidates if 1 <= e <= k - 2]

    out: list[list[int]] = []
    for extra_blocks in range(segments):
        for ends in combinations(candidates, extra_blocks):
            payments = _blocks_to_payments(ends, k, total, floors)
            if payments is not None:
                out.append(payments)
    return out


def candidate_payments(rules: CreditorRules, k: int, total: int) -> list[list[int]]:
    """All payment vectors to consider for a given payment count."""
    floors = payment_floors(rules, k)
    if sum(floors) > total:
        return []

    if rules.even_pays:
        payments = even_split(total, k)
        ok = all(p >= f for p, f in zip(payments, floors))
        return [payments] if ok else []

    if rules.is_ballooning_allowed:
        # Floors everywhere, remainder in the final payment. This vector
        # minimises *every* prefix sum simultaneously, so if it does not fit
        # the cash, no vector for this k does.
        head = floors[:-1]
        last = total - sum(head)
        return [head + [last]] if last >= floors[-1] else []

    return staircase_candidates(rules, k, total, floors)


def classify_shape(rules: CreditorRules, payments: list[int]) -> str:
    """Report the shape actually produced.

    "Even" is tested against ``even_split`` rather than ``len(set(...)) == 1``:
    §7 defines an even schedule as "as equal as possible", so ``[8333, 8334]``
    is even in exactly the way ``[8333, 8333]`` is. Testing for one distinct
    value would make the reported shape hinge on whether ``k`` happens to divide
    ``offer_total`` — the same schedule reported "even" for a total of $100 over
    two payments and "staircase" for $100.01.

    A balloon is then a final payment that jumps by more than a rounding cent.
    An as-equal-as-possible level has ``max - min <= 1``, so a gap of two cents
    or more is what separates a genuine balloon — §2's "final payment absorbing
    the entire remaining balance" — from the ``+1`` cent remainder of a level.
    """
    if rules.even_pays or payments == even_split(sum(payments), len(payments)):
        return "even"
    if (
        rules.is_ballooning_allowed
        and len(payments) >= 2
        and payments[-1] - payments[-2] >= 2
    ):
        return "balloon"
    return "staircase"


# ---------------------------------------------------------------------------
# Plan search
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    dates: list[date]
    payments: list[int]      # per cadence index, 0 on fee-only dates
    fees: list[int]          # per cadence index
    banks: list[int]         # per cadence index
    balances: list[int]      # per cadence index, after credits and all debits
    k: int
    shape: str
    key: tuple


def _evaluate_vector(
    cadence: list[date],
    caps: CashCaps,
    rules: CreditorRules,
    payments: list[int],
    fee_total: int,
) -> Plan | None:
    """Place the fee as early as possible against a fixed payment vector."""
    m = len(cadence)
    k = len(payments)
    bank = rules.bank_fee_cents

    # Cumulative creditor payments + bank fees at each cadence date.
    cum_pb: list[int] = []
    running = 0
    for i in range(m):
        if i < k:
            running += payments[i] + bank
        cum_pb.append(running)

    # Headroom for fee at cadence date i: the cash left over cap[i] once this
    # date's committed payments and bank fees are taken. The fee we hold at date
    # i is still held at every later date, while cum_pb keeps growing, so the
    # binding limit is the *suffix minimum of the difference* — not the suffix
    # minimum of cap alone. Folding the suffix min into cap on its own would let
    # the greedy take fee out of headroom that a later creditor payment needs,
    # and then wrongly report the whole offer infeasible.
    headroom = [caps.cap[i] - cum_pb[i] for i in range(m)]
    for i in range(m - 2, -1, -1):
        headroom[i] = min(headroom[i], headroom[i + 1])

    # `headroom` is a suffix minimum and therefore non-decreasing, which is what
    # makes this one-pass greedy lexicographically optimal: taking the most we
    # can now never strands fee that a later date could have carried.
    fees = [0] * m
    fee_cum: list[int] = []
    collected = 0
    for i in range(m):
        room = headroom[i] - collected
        if room < 0:
            return None
        take = min(fee_total - collected, room)
        fees[i] = take
        collected += take
        fee_cum.append(collected)
    if collected != fee_total:
        return None  # fee cannot be fully collected by the horizon

    banks = [bank if i < k else 0 for i in range(m)]
    balances = [caps.balance_at[i] - cum_pb[i] - fee_cum[i] for i in range(m)]

    # Objective: maximise collected-fee-to-date lexicographically. Tie-break by
    # deferring creditor money as long as possible (matters when the fee is 0,
    # or once the fee is fully collected), then by the fewest payments.
    key = (tuple(-f for f in fee_cum), tuple(cum_pb), k)

    return Plan(
        dates=list(cadence),
        payments=[payments[i] if i < k else 0 for i in range(m)],
        fees=fees,
        banks=banks,
        balances=balances,
        k=k,
        shape=classify_shape(rules, payments),
        key=key,
    )


def solve(
    client: Client,
    offer: Offer,
    rules: CreditorRules,
    extra: tuple[LedgerEntry, ...] | list[LedgerEntry] = (),
) -> Plan | None:
    """Best plan under the objective, or None if no valid schedule exists."""
    cadence = cadence_dates(client, offer)
    if not cadence:
        return None

    caps = cash_caps(client, cadence, extra)
    if not caps.ok_before_first:
        # The committed ledger already goes negative before we schedule
        # anything; no choice of ours can undo that.
        return None

    total = offer_total_cents(offer)
    fee_total = program_fee_cents(offer, rules)
    if total < 0 or fee_total < 0:
        return None

    k_max = min(rules.max_payments, rules.max_terms, len(cadence))
    best: Plan | None = None
    for k in range(1, k_max + 1):
        for payments in candidate_payments(rules, k, total):
            plan = _evaluate_vector(cadence, caps, rules, payments, fee_total)
            if plan is not None and (best is None or plan.key < best.key):
                best = plan
    return best


# ---------------------------------------------------------------------------
# Part 2: minimum additional funds
# ---------------------------------------------------------------------------

def _smallest_feasible(predicate) -> int | None:
    """Smallest positive amount satisfying a monotone predicate.

    Injecting money only raises the cash curve, so feasibility is monotone in
    the amount: doubling finds a bracket, then bisection finds the threshold.
    """
    high = 1
    while not predicate(high):
        high *= 2
        if high > MAX_FUNDING_SEARCH:
            return None
    low = 0  # known infeasible: this is only called when 0 does not work
    while low + 1 < high:
        mid = (low + high) // 2
        if predicate(mid):
            high = mid
        else:
            low = mid
    return high


def lump_sum_date(client: Client) -> date:
    """Where we place the lump: the earliest date we are allowed to modify.

    An earlier lump is weakly more useful (it is available at every date a later
    one would be), so the smallest L is always attained at the earliest date.
    """
    return client.as_of_date + timedelta(days=1)


def min_lump_sum(client: Client, offer: Offer, rules: CreditorRules) -> tuple[int | None, date]:
    when = lump_sum_date(client)
    if when > client.last_draft_date:
        return None, when

    def feasible(amount: int) -> bool:
        entry = LedgerEntry(when, amount, "credit")
        return solve(client, offer, rules, extra=[entry]) is not None

    return _smallest_feasible(feasible), when


def future_drafts(client: Client) -> list[LedgerEntry]:
    """The drafts we can still raise: ledger credits dated after as_of_date."""
    return [e for e in client.ledger if e.type == "credit" and e.date > client.as_of_date]


def min_monthly_increment(
    client: Client, offer: Offer, rules: CreditorRules
) -> tuple[int | None, int]:
    drafts = future_drafts(client)
    if not drafts:
        return None, 0

    def feasible(amount: int) -> bool:
        extra = [LedgerEntry(d.date, amount, "credit") for d in drafts]
        return solve(client, offer, rules, extra=extra) is not None

    return _smallest_feasible(feasible), len(drafts)
