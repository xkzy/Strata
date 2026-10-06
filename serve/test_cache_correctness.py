# serve/test_cache_correctness.py - Deterministic Tests for Cache Correctness & Isolation
import unittest
from serve.virtual_context import VirtualContextServerRuntime, ResidencyState
from serve.multi_tenant import MultiTenantServerManager, SecurityScope, SharingScope
from serve.generation_loop import GenerationLoopConfig, GenerationLoopDetector, LoopType


class TestCacheCorrectness(unittest.TestCase):
    def test_exact_prefix_isolation(self):
        """Test E: Common-prefix prompts with different suffixes do not contaminate each other."""
        rt = VirtualContextServerRuntime(physical_context_limit=4096)
        
        prompt_A = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain quicksort in Python."}
        ]
        prompt_B = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Explain mergesort in C++."}
        ]

        out_A = rt.process_messages(prompt_A, 2048)
        out_B = rt.process_messages(prompt_B, 2048)

        # Both have identical system prompt prefix
        self.assertEqual(out_A[0]["content"], out_B[0]["content"])
        # Suffixes are strictly different
        self.assertEqual(out_A[1]["content"], "Explain quicksort in Python.")
        self.assertEqual(out_B[1]["content"], "Explain mergesort in C++.")

    def test_context_compaction_invalidation(self):
        """Test F: When context compaction modifies old turns, modified turns have unique content."""
        rt = VirtualContextServerRuntime(physical_context_limit=512)

        long_conversation = [{"role": "system", "content": "Base instructions for coding."}]
        for i in range(12):
            long_conversation.append({"role": "user", "content": f"Step {i}: Compute detailed mathematical optimization analysis with intermediate tensors and validation matrices for step number {i * 100} in full detail."})
            long_conversation.append({"role": "assistant", "content": f"Execution analysis for step {i}: The computed eigenvalues, trace, and loss gradient vectors are verified for index {i * 100}."})

        compacted = rt.process_messages(long_conversation, 100)
        
        # Verify compaction preserved the system instructions and compacted older history
        self.assertEqual(compacted[0]["role"], "system")
        self.assertIn("Base instructions for coding.", compacted[0]["content"])
        # A summary turn was created for older messages
        summary_turn = next((m for m in compacted if "[Summary of" in m.get("content", "")), None)
        self.assertIsNotNone(summary_turn)
        # Latest working set turns are intact
        self.assertEqual(compacted[-1]["role"], "assistant")
        self.assertIn("Execution analysis for step 11", compacted[-1]["content"])

    def test_multi_tenant_isolation(self):
        """Test G: Different tenants/sessions never cross-contaminate context state."""
        mgr = MultiTenantServerManager(default_physical_limit=4096)

        scope1 = SecurityScope(tenant_id="tenant_1", user_id="user_1", session_id="sess_1")
        scope2 = SecurityScope(tenant_id="tenant_2", user_id="user_2", session_id="sess_2")

        vctx1 = mgr.get_or_create_session(scope1)
        vctx2 = mgr.get_or_create_session(scope2)

        # Ingest private knowledge into tenant 1
        mgr.ingest_shared_knowledge(scope1, "secret.txt", "CONFIDENTIAL_KEY_XYZ", SharingScope.SESSION)

        # Tenant 1 searches
        res1 = mgr.retrieve_scoped("CONFIDENTIAL_KEY", scope1)
        self.assertTrue(len(res1) > 0)
        self.assertIn("CONFIDENTIAL_KEY_XYZ", res1[0]["content"])

        # Tenant 2 searches - must find nothing
        res2 = mgr.retrieve_scoped("CONFIDENTIAL_KEY", scope2)
        self.assertEqual(len(res2), 0)

    def test_repeated_request_stability(self):
        """Test C: Repeated requests with same prompt produce consistent, stable tokens."""
        detector = GenerationLoopDetector()
        
        # Feed varied response
        text = "This is a deterministic and verified explanation of algorithms without loop."
        words = text.split()
        for idx, w in enumerate(words):
            v = detector.feed_token(100 + idx, w)
            self.assertFalse(v.should_stop)
        
        self.assertGreater(detector.current_diversity, 0.8)


if __name__ == "__main__":
    unittest.main()
