"""Tactical analysis + anti-game-plan generator (v1.0).

Heuristic, data-driven tactical profiling from the match summary.
NOT a fitted ML model — every finding cites the underlying numbers.
Given a team's profile, it finds structural vulnerabilities and turns
each one into a concrete counter-instruction.
"""
from __future__ import annotations


def _flank(y):
    if y < -11:
        return "left"
    if y > 11:
        return "right"
    return "center"


def tactical_profile(summary):
    """Build a tactical profile per team from the summary."""
    profile = {}
    attack_dirs = {}
    # infer attack directions from shots (same logic as events)
    for t in ("A", "B"):
        shots = [s for s in summary.get("shots", []) if s.get("team") == t]
        dirs = [s.get("goal") for s in shots if s.get("goal")]
        if dirs:
            attack_dirs[t] = 1 if dirs.count("+x") >= dirs.count("-x") else -1
        else:
            attack_dirs[t] = 1 if t == "A" else -1

    for t in ("A", "B"):
        d = attack_dirs[t]
        # attacking zones from shots + progressive pass receptions
        zones = {"left": 0, "center": 0, "right": 0}
        for s in summary.get("shots", []):
            if s.get("team") == t:
                zones[_flank(s["y"])] += 1
        for p in summary.get("progressive_passes", []):
            if p["team"] == t and "to_xy" in p:
                zones[_flank(p["to_xy"][1])] += 1
        total_z = sum(zones.values()) or 1

        # defensive line: average x of the 4 deepest outfield positions
        # (deepest = smallest x*d, i.e. furthest from opponent goal)
        pos = summary.get("shape", {}).get("positions", {}).get(t, [])
        deep = sorted(pos, key=lambda xy: xy[0] * d)[:4]
        line_h = (sum(x for x, _ in deep) / len(deep)) if deep else None

        # key distributor: most passes made
        players = summary.get("players", {})
        mates = {tid: s for tid, s in players.items() if s["team"] == t}
        total_p = sum(s["passes_made"] for s in mates.values()) or 1
        if mates:
            kid, ks = max(mates.items(),
                          key=lambda kv: kv[1]["passes_made"])
            share = ks["passes_made"] / total_p
        else:
            kid, share = None, 0.0

        # turnovers in own half (x*d < 0)
        own_turn = 0
        for p in summary.get("passes", {}).get("list", []):
            if p.get("team") == t and "from_xy" in p:
                # turnovers aren't in pass_list; use team turnovers proxy:
                pass
        # use completed-pass origins as a proxy is weak; instead count
        # turnovers via the summary counters and flag high-press weakness
        # from carry/turnover ratio
        turnovers = summary.get("passes", {}).get("turnovers", {}).get(t, 0)
        completed = summary.get("passes", {}).get("completed", {}).get(t, 0)

        # shot quality
        tshots = [s for s in summary.get("shots", []) if s.get("team") == t]
        avg_xg = (sum(s.get("xg", 0) for s in tshots) / len(tshots)
                  if tshots else 0.0)

        profile[t] = {
            "attack_dir": d,
            "zones": {k: round(v / total_z, 2) for k, v in zones.items()},
            "dominant_flank": max(zones, key=zones.get),
            "flank_share": round(max(zones.values()) / total_z, 2),
            "defensive_line_x": round(line_h, 1) if line_h is not None else None,
            "high_line": (line_h * d) > -5.0 if line_h is not None else False,
            "key_distributor": kid,
            "distributor_share": round(share, 2),
            "turnovers": turnovers,
            "completed": completed,
            "turnover_rate": round(turnovers / (turnovers + completed), 2)
            if (turnovers + completed) else 0.0,
            "avg_xg_per_shot": round(avg_xg, 3),
            "n_shots": len(tshots),
        }
    return profile


def find_vulnerabilities(profile):
    """Turn a tactical profile into a list of exploitable weaknesses."""
    vulns = []
    for t, p in profile.items():
        opp = "B" if t == "A" else "A"
        # 1. high defensive line -> balls in behind
        if p["high_line"] and p["defensive_line_x"] is not None:
            vulns.append({
                "team": t, "severity": "high",
                "type": "high_line",
                "finding": f"Team {t} holds a high defensive line "
                           f"(avg {p['defensive_line_x']:.0f}m).",
                "exploit": "Play direct balls in behind their defence; "
                           "run channels with pace.",
            })
        # 2. over-reliance on one distributor
        if p["distributor_share"] >= 0.4 and p["key_distributor"] is not None:
            vulns.append({
                "team": t, "severity": "high",
                "type": "key_man",
                "finding": f"Team {t} routes {p['distributor_share']:.0%} of "
                           f"passes through #{p['key_distributor']}.",
                "exploit": f"Man-mark #{p['key_distributor']} and press "
                           f"their first touch to kill Team {t}'s build-up.",
            })
        # 3. one-sided attack -> exposed opposite flank
        if p["flank_share"] >= 0.6:
            weak = {"left": "right", "right": "left",
                    "center": "wide areas"}[p["dominant_flank"]]
            vulns.append({
                "team": t, "severity": "medium",
                "type": "flank_exposure",
                "finding": f"Team {t} focuses {p['flank_share']:.0%} of "
                           f"attacks down the {p['dominant_flank']}.",
                "exploit": f"They leave {weak} exposed on transitions — "
                           f"switch play quickly to attack there.",
            })
        # 4. sloppy in possession
        if p["turnover_rate"] >= 0.3 and (p["turnovers"] + p["completed"]) >= 5:
            vulns.append({
                "team": t, "severity": "medium",
                "type": "pressable",
                "finding": f"Team {t} turns it over "
                           f"{p['turnover_rate']:.0%} of the time.",
                "exploit": f"Press Team {t} high — they crack under "
                           f"pressure and give the ball away.",
            })
        # 5. low shot quality
        if p["n_shots"] >= 3 and p["avg_xg_per_shot"] < 0.08:
            vulns.append({
                "team": t, "severity": "low",
                "type": "poor_chance_creation",
                "finding": f"Team {t} averages {p['avg_xg_per_shot']:.3f} xG "
                           f"per shot — mostly hopeful efforts.",
                "exploit": f"Stay compact and let Team {t} shoot from "
                           f"distance; don't dive in.",
            })
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(vulns, key=lambda v: order[v["severity"]])


def anti_game_plan(summary, vs_team="B"):
    """Generate a counter game plan against vs_team.

    Returns (profile, vulnerabilities, plan) where plan is a list of
    concrete instructions.
    """
    profile = tactical_profile(summary)
    vulns = [v for v in find_vulnerabilities(profile) if v["team"] == vs_team]
    plan = []
    for v in vulns:
        plan.append(f"[{v['severity'].upper()}] {v['exploit']}")
    # shape-based instruction
    p = profile[vs_team]
    if p["attack_dir"] == 1:
        plan.append("They attack left-to-right: overload the midfield "
                    "on their strong side early.")
    # general
    if not plan:
        plan.append("No clear structural weakness found — stay compact, "
                    "win second balls, and be patient.")
    return profile, vulns, plan
