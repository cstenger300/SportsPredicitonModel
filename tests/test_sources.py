import pandas as pd
import pytest
import requests
from nfl_model.sources import normalize_odds, get_json, SourceUnavailable, fetch_lineups, map_quarterback_ids


def test_odds_parser_keeps_prices_lines_and_provider_timestamps():
    payload = [{"home_team": "Los Angeles Rams", "away_team": "Buffalo Bills", "commence_time": "2026-10-13T00:15:00Z",
                "bookmakers": [{"key": "book", "markets": [{"key": "spreads", "last_update": "2026-10-12T16:00:00Z",
                "outcomes": [{"name": "Los Angeles Rams", "point": -3.5, "price": -110},
                             {"name": "Buffalo Bills", "point": 3.5, "price": -110}]}]}]}]
    parsed = normalize_odds(payload, {"game_id": "2026_05_BUF_LA", "home_team": "LA", "away_team": "BUF"}, "2026-10-12T16:01:00Z")
    assert set(parsed.side) == {"home", "away"}
    assert parsed.loc[parsed.side.eq("home"), "point"].iloc[0] == -3.5
    assert parsed.source_updated_at.lt(parsed.observed_at).all()


def test_authenticated_request_errors_do_not_expose_key_or_url(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("request containing confidential-token")
    monkeypatch.setattr(requests, "get", fail)
    with pytest.raises(SourceUnavailable) as caught:
        get_json("https://api.the-odds-api.com", params={"apiKey": "confidential-token"})
    assert "confidential-token" not in str(caught.value)
    assert "api.the-odds-api.com" not in str(caught.value)


def test_actual_depthchart_schema_and_id_mapping(monkeypatch, tmp_path):
    board = {"events": [{"date": "2026-10-13T00:15Z", "status": {"type": {"state": "pre"}},
                          "competitions": [{"competitors": [{"team": {"abbreviation": "BUF", "id": "2"}},
                                                              {"team": {"abbreviation": "LAR", "id": "14"}}]}]}]}
    depth = {"depthchart": [{"positions": {"qb": {"position": {"abbreviation": "QB"},
                                                   "athletes": [{"id": "123", "displayName": "Listed quarterback"}]}}}]}
    def fake(url, **kwargs):
        if url.endswith("depthcharts"):
            return depth
        if url.endswith("roster"):
            return {"athletes": []}
        return board
    monkeypatch.setattr("nfl_model.sources.get_json", fake)
    context = fetch_lineups({"game_id": "2026_05_BUF_LA", "gameday": "2026-10-12", "home_team": "LA", "away_team": "BUF"}, tmp_path)
    roster = pd.DataFrame({"team": ["BUF", "LA"], "position": ["QB", "QB"], "espn_id": ["123", "123"], "gsis_id": ["buf-qb", "la-qb"]})
    context = map_quarterback_ids(context, roster)
    assert context["teams"]["BUF"]["expected_qb_gsis_id"] == "buf-qb"
    assert context["teams"]["LA"]["expected_qb"]["name"] == "Listed quarterback"
    assert context["lineups_reviewed"] is False


def test_roster_refresh_preserves_cache_on_invalid_response(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace
    from nfl_model.sources import load_roster
    initial = pd.DataFrame({"team": ["BUF"], "position": ["QB"], "espn_id": ["123"], "gsis_id": ["old-id"]})
    target = tmp_path / "roster_2026.parquet"
    initial.to_parquet(target)
    original = target.read_bytes()
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(ok=True, status_code=200, content=b"not a parquet file"))
    with pytest.raises(SourceUnavailable, match="previous cache retained"):
        load_roster(2026, tmp_path, refresh=True)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.part"))
    updated = initial.copy()
    updated["gsis_id"] = "new-id"
    buffer = io.BytesIO()
    updated.to_parquet(buffer)
    monkeypatch.setattr(requests, "get", lambda *a, **kw: SimpleNamespace(ok=True, status_code=200, content=buffer.getvalue()))
    assert load_roster(2026, tmp_path, refresh=True).gsis_id.iloc[0] == "new-id"
    archives = list((tmp_path / "roster_archive").glob("*.parquet"))
    assert len(archives) == 1 and archives[0].read_bytes() == original
