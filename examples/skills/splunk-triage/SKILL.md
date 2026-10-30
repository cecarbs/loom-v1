---
name: splunk-triage
description: Use when investigating errors, latency or outages with Splunk data.
---
# Splunk triage

Work in `execute`. Keep raw results in kernel variables; print only small summaries.

1. Run `search_tools("splunk")` to see the available functions, and `splunk.list_indexes()`
   to see which indexes you may query. Always name an index in SPL.
2. Pin down the incident window first (ask the user if it isn't given). Use explicit
   `earliest` / `latest` (e.g. `"-6h"`, ISO timestamps) on every search.
3. Start with aggregates, never raw events:
   - `index=web status>=500 | stats count by host, status, uri | sort -count`
   - `index=web | timechart span=5m count by status` to see when it started
   - `index=web | stats perc95(duration_ms) as p95 by uri | sort -p95` for latency
   Keep `max_rows` small (≤ 200) and store results: `errors = splunk.search(...)`.
4. Narrow down: compare the incident window with the same window a day earlier; break the
   worst bucket down by host, then by uri or error message.
5. Only then pull a few raw events for the top offender
   (`... | head 20 | table _time host uri status message`) to quote exact messages.
6. Look for a trigger: deploys, config changes or restarts just before the start
   (`index=app (deploy OR restart) | table _time host message`).
7. Report: start time, blast radius (hosts, endpoints, % of requests), the likely cause with
   the evidence (searches and numbers), and open questions. Offer a saved search for the
   symptom (`splunk.create_saved_search`, which needs approval) instead of creating one.
