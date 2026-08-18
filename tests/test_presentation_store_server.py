from pathlib import Path

from stroke_screening.presentation_server import create_app
from stroke_screening.presentation_store import PresentationStore


class MemoryKeys:
    def __init__(self): self.values = {}
    def save(self, profile, key): self.values[profile] = key
    def load(self, profile): return self.values[profile]
    def delete(self, profile): self.values.pop(profile, None)


class FakeLLM:
    def status(self): return {"available": True, "installed": True, "model": "fake"}
    def complete_json(self, **_kwargs): return {"strengths": [], "improvements": [], "insufficient_data": []}


class FakeRecorder:
    pass


def test_encrypted_profile_round_trip_and_csrf(tmp_path: Path):
    keys = MemoryKeys()
    store = PresentationStore(data_dir=tmp_path / "data", key_store=keys)
    app = create_app(store=store, llm=FakeLLM(), recorder=FakeRecorder(), testing=True)
    with app.test_client() as client:
        bootstrap = client.get("/api/bootstrap")
        csrf = bootstrap.get_json()["csrf_token"]
        denied = client.post("/api/profiles", json={"name": "Tanav"})
        assert denied.status_code == 403
        created = client.post(
            "/api/profiles", json={"name": "Tanav"}, headers={"X-CSRF-Token": csrf}
        )
        assert created.status_code == 201
        profile_id = created.get_json()["profile"]["id"]
        loaded = client.get(f"/api/bootstrap?profile={profile_id}").get_json()
        assert loaded["profile"] == {"id": profile_id, "name": "Tanav"}
        assert loaded["calibration"]["stage"] == "record_baseline"
        encrypted = next((tmp_path / "data").glob("*.presentcoach")).read_bytes()
        assert b"Tanav" not in encrypted
