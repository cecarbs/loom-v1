"""Example loom extension. Copy it to ~/.loom/extensions/ (or a trusted workspace's
.loom/extensions/) to use it."""

from loom.extensions import Block, BrokerCall, ExtensionAPI


def register(api: ExtensionAPI) -> None:
    @api.on("before_broker_call")
    async def guard(call: BrokerCall):
        if call.effect != "read" and "prod" in str(call.args):
            return Block("Writes mentioning prod are disabled by policy.")

    @api.command("incident", help="investigate an incident with the splunk-triage skill")
    async def incident(args: str, ctx):
        await ctx.submit(f"Load the splunk-triage skill and investigate: {args}")
