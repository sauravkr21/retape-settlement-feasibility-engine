"""Public entry point: ``evaluate_offer``.

The output dataclasses are unchanged from the scaffolding. The modelling lives
in ``feasibility.solver``; see README.md, "Implementation notes", for the
objective and the shape interpretations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

from feasibility.models import Client, CreditorRules, Offer, offer_total_cents
from feasibility.money import pct_of_cents
from feasibility.solver import min_lump_sum, min_monthly_increment, solve


@dataclass
class ScheduleRow:
    date: date
    creditor_payment_cents: int
    program_fee_cents: int
    bank_fee_cents: int
    balance_cents: int


@dataclass
class FundsOption:
    amount_cents: int
    within_guardrail: bool
    reason: str
    # lump-sum only:
    date: date | None = None
    # monthly-increment only:
    num_drafts: int | None = None


@dataclass
class AdditionalFunds:
    lump_sum: FundsOption
    monthly_increment: FundsOption


@dataclass
class Result:
    feasible: bool
    # One of "even", "staircase", or "balloon" — the shape your solution produced
    # (driven by the creditor flags). None when infeasible.
    pay_shape_used: str | None = None
    schedule: list[ScheduleRow] | None = None
    additional_funds: AdditionalFunds | None = None

    def to_dict(self) -> dict:
        out: dict = {"feasible": self.feasible, "pay_shape_used": self.pay_shape_used}
        out["schedule"] = (
            [
                {
                    "date": r.date.isoformat(),
                    "creditor_payment_cents": r.creditor_payment_cents,
                    "program_fee_cents": r.program_fee_cents,
                    "bank_fee_cents": r.bank_fee_cents,
                    "balance_cents": r.balance_cents,
                }
                for r in self.schedule
            ]
            if self.schedule is not None
            else None
        )
        if self.additional_funds is None:
            out["additional_funds"] = None
        else:
            def opt(o: FundsOption, *, lump: bool) -> dict:
                d = {
                    "amount_cents": o.amount_cents,
                    "within_guardrail": o.within_guardrail,
                    "reason": o.reason,
                }
                # Each option carries exactly one extra field, and carries it
                # unconditionally. The key set must not vary with the outcome:
                # emitting "date" only when there is one would make
                # additional_funds["lump_sum"]["date"] raise on the case where
                # no lump sum helps, rather than reading null.
                if lump:
                    d["date"] = o.date.isoformat() if o.date is not None else None
                else:
                    d["num_drafts"] = o.num_drafts
                return d

            out["additional_funds"] = {
                "lump_sum": opt(self.additional_funds.lump_sum, lump=True),
                "monthly_increment": opt(
                    self.additional_funds.monthly_increment, lump=False
                ),
            }
        return out


def evaluate_offer(client: Client, offer: Offer, rules: CreditorRules) -> Result:
    """Evaluate a single offer. See ASSIGNMENT.md for the full specification.

    Return a Result with feasible=True and a schedule when the offer fits, or
    feasible=False with additional_funds (minimum lump sum AND minimum monthly
    increment) when it does not.
    """
    plan = solve(client, offer, rules)
    if plan is not None:
        schedule = [
            ScheduleRow(
                date=plan.dates[i],
                creditor_payment_cents=plan.payments[i],
                program_fee_cents=plan.fees[i],
                bank_fee_cents=plan.banks[i],
                balance_cents=plan.balances[i],
            )
            for i in range(len(plan.dates))
            # Emit the dates we actually use: those carrying a creditor payment
            # (and therefore a bank fee) and any fee-only date.
            if i < plan.k or plan.fees[i] > 0
        ]
        return Result(
            feasible=True,
            pay_shape_used=plan.shape,
            schedule=schedule,
            additional_funds=None,
        )

    return Result(
        feasible=False,
        pay_shape_used=None,
        schedule=None,
        additional_funds=_additional_funds(client, offer, rules),
    )


def _blocked_reason(client: Client, funding: str) -> str:
    """Why no amount of ``funding`` can rescue this offer.

    A negative opening balance is called out separately: it *is* a cash
    shortfall, so blaming dates or floors would be actively misleading, and no
    credit dated after ``as_of_date`` can lift a balance that already went
    negative on it.
    """
    if client.current_balance_cents < 0:
        return (
            f"the account already stands at {client.current_balance_cents} cents "
            f"on {client.as_of_date.isoformat()}; no {funding} dated after that "
            f"can repair a balance that is negative before the schedule starts"
        )
    return (
        f"no {funding} makes this offer feasible: the schedule is blocked by a "
        f"non-cash constraint (dates, floors, or segment limits)"
    )


def _additional_funds(
    client: Client, offer: Offer, rules: CreditorRules
) -> AdditionalFunds:
    """The two independent minima of ASSIGNMENT.md §8, plus their guardrails."""
    lump_cap = pct_of_cents(0.65, offer_total_cents(offer))
    increment_cap = max(10000, pct_of_cents(0.40, client.draft_amount_cents))

    amount, when = min_lump_sum(client, offer, rules)
    if amount is None:
        lump = FundsOption(
            amount_cents=0,
            within_guardrail=False,
            reason=_blocked_reason(client, "lump sum"),
            date=None,
        )
    elif amount > lump_cap:
        lump = FundsOption(
            amount_cents=amount,
            within_guardrail=False,
            reason=(
                f"lump sum {amount} exceeds the cap of {lump_cap} "
                f"(65% of the {offer_total_cents(offer)} offer total)"
            ),
            date=when,
        )
    else:
        lump = FundsOption(
            amount_cents=amount, within_guardrail=True, reason="", date=when
        )

    amount, num_drafts = min_monthly_increment(client, offer, rules)
    if amount is None:
        increment = FundsOption(
            amount_cents=0,
            within_guardrail=False,
            reason=(
                "no future drafts remain to raise"
                if num_drafts == 0
                else _blocked_reason(client, "monthly increment")
            ),
            num_drafts=num_drafts,
        )
    elif amount > increment_cap:
        increment = FundsOption(
            amount_cents=amount,
            within_guardrail=False,
            reason=(
                f"monthly increment {amount} exceeds the cap of {increment_cap} "
                f"(the greater of 10000 and 40% of the "
                f"{client.draft_amount_cents} draft)"
            ),
            num_drafts=num_drafts,
        )
    else:
        increment = FundsOption(
            amount_cents=amount,
            within_guardrail=True,
            reason="",
            num_drafts=num_drafts,
        )

    return AdditionalFunds(lump_sum=lump, monthly_increment=increment)
