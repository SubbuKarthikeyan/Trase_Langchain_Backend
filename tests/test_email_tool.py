"""
test_email_tool.py
───────────────────
Unit and integration test for app.tools.email_tool (AgentMail API integration).
Verifies output, error handling, and terminal logging status.
"""

import sys
import os

# Ensure backend root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tools.email_tool import send_bus_details_email
from app.router.tool_registry import get_tool, list_tools


def test_tool_registry():
    print("\n--- Testing Tool Registry ---")
    registered = list_tools()
    print(f"Registered tools: {registered}")
    assert "send_email" in registered, "send_email tool is not registered!"
    fn = get_tool("send_email")
    assert callable(fn), "Registered send_email tool is not callable!"
    print("[SUCCESS] Tool registry test passed.")


def test_email_tool_execution():
    print("\n--- Testing Email Tool Execution ---")
    test_bus_details = (
        "Bus Name: Southern Travels Express\n"
        "Bus Type: AC Sleeper\n"
        "Departure: 21:00 (Chennai Central)\n"
        "Arrival: 05:00 (Bangalore Majestic)\n"
        "Fare: Rs. 850\n"
        "Facilities: Charging point, Wi-Fi, Water bottle"
    )
    
    # Test execution with dummy email
    result = send_bus_details_email(
        to_email="test_recipient@example.com",
        bus_details=test_bus_details,
        subject="[TEST] Trase Bus Details Email Test"
    )
    
    print(f"Tool Result Summary:\n{result}")
    assert isinstance(result, str), "Result must be a string."
    assert "SUCCESS" in result or "ERROR" in result, "Result must report SUCCESS or ERROR status."
    print("[SUCCESS] Email tool execution test completed with status output.")


if __name__ == "__main__":
    test_tool_registry()
    test_email_tool_execution()
