from typing import List, Dict


def render_timeline(timeline: List[Dict]) -> None:
    """
    Simple console-based analyst view.
    (Replace with API / UI later)
    """

    for entry in timeline:
        print("=" * 80)
        print(f"🆔 Incident ID: {entry['incident_id']}")
        print(f"🕒 Timestamp: {entry['timestamp']}")
        print(f"🌍 Environment: {entry['environment']}")
        print(
            f"🤖 Model: {entry['model']['name']} "
            f"(v{entry['model']['version']} - {entry['model']['stage']})"
        )
        print(f"⚠️ Trigger: {entry['trigger_reason']}")
        print(f"🔁 Retrain Triggered: {entry['retrain_triggered']}")
        print("\n📄 Summary:")
        print(entry["summary"])
        print("\n🧠 Explanation:")
        print(entry["explanation"])
        print("=" * 80)
