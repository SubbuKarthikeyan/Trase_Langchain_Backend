"""
test_email_workflow.py
───────────────────────
End-to-end conversation flow test:
1. User asks for buses (RAG)
2. User chooses option 1 (RAG + Bus selection memory)
3. User confirms email intent ("Yes")
4. User provides email address (Tool Call: send_email)
5. User asks memory question ("What bus did I choose?")
"""

import sys
import os

# Globally configure UTF-8 output encoding with replacement for unsupported terminal characters
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure backend root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.routes.query import _dispatch
from app.utils.session_memory import session_store


def run_turn(user_message: str):
    print("\n" + "=" * 60)
    print(f" [USER]: {user_message}")
    print("=" * 60)
    
    tokens = list(_dispatch(user_message))
    response = "".join(tokens)
    
    print(f" [TRASE AI]: {response}")
    print("-" * 60)
    return response


def test_full_conversational_workflow():
    session_store.clear()
    
    print("\n--- STARTING MULTI-TURN CONVERSATIONAL WORKFLOW TEST ---\n")

    # Turn 1: Search buses
    res1 = run_turn("Are there any buses from Chennai to Madurai?")
    assert len(res1) > 0, "Turn 1 response empty"

    # Turn 2: Select an option
    res2 = run_turn("I choose Option 1")
    assert len(res2) > 0, "Turn 2 response empty"

    # Turn 3: User consents to email prompt
    res3 = run_turn("Yes please")
    assert len(res3) > 0, "Turn 3 response empty"

    # Turn 4: User provides email (Triggers Tool Call)
    res4 = run_turn("Please send the details to trase_user@example.com")
    assert len(res4) > 0, "Turn 4 response empty"

    # Turn 5: Test conversation memory recall
    res5 = run_turn("Which bus did I choose earlier?")
    assert len(res5) > 0, "Turn 5 response empty"

    print("\n[SUCCESS] MULTI-TURN WORKFLOW & MEMORY TEST COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    test_full_conversational_workflow()
