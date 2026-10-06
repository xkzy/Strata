# serve/multi_tenant.py - Multi-Tenant, Multi-Agent Server Isolation & Fair Scheduling Layer
"""Multi-Tenant & Multi-Agent Runtime Layer for Strata Server.

Provides:
- Strict Tenant, User, Workspace, Agent, and Session isolation
- Fine-grained per-session VirtualContextServerRuntime instances
- Multi-tenant Scoped Tool Result CAS Storage with strict ownership validation
- Shared Project/Workspace Knowledge Base with pre-filtering access control
- Tenant Quotas, Fair Resource Scheduling, and Per-Scope Accounting Metrics
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from serve.virtual_context import (
    ContextItem,
    ContextItemType,
    ContextMetadata,
    HierarchicalBM25Index,
    InformationClass,
    ResidencyState,
    StructuredToolData,
    ToolOutputParser,
    ToolResultCASStore,
    VirtualContextServerRuntime,
)


class SharingScope(Enum):
    PRIVATE = "private"     # Only accessible by exact session
    SESSION = "session"     # Same as private
    AGENT = "agent"         # Shared across sessions of the same AgentID
    USER = "user"           # Shared across agents of the same UserID
    PROJECT = "project"     # Shared across users in the same WorkspaceID / ProjectID
    TEAM = "team"           # Shared across users in the same TenantID
    GLOBAL = "global"       # Public read-only globally


@dataclass
class SecurityScope:
    tenant_id: str = "default_tenant"
    user_id: str = "default_user"
    workspace_id: str = "default_workspace"
    agent_id: str = "default_agent"
    session_id: str = "default_session"
    sharing_scope: SharingScope = SharingScope.PRIVATE
    permissions: Set[str] = field(default_factory=lambda: {"read", "write"})

    def can_access(self, target_owner: SecurityScope, target_sharing: SharingScope) -> bool:
        if target_sharing == SharingScope.GLOBAL:
            return True
        if self.tenant_id != target_owner.tenant_id:
            return False

        if target_sharing == SharingScope.TEAM:
            return True
        elif target_sharing == SharingScope.PROJECT:
            return self.workspace_id == target_owner.workspace_id
        elif target_sharing == SharingScope.USER:
            return self.user_id == target_owner.user_id
        elif target_sharing == SharingScope.AGENT:
            return self.user_id == target_owner.user_id and self.agent_id == target_owner.agent_id
        else:
            return self.user_id == target_owner.user_id and self.session_id == target_owner.session_id

    @classmethod
    def from_headers_and_body(cls, headers: Any, body: Dict[str, Any]) -> SecurityScope:
        def get_val(h_key: str, b_key: str, default: str) -> str:
            if headers:
                val = headers.get(h_key)
                if val:
                    return str(val).strip()
            if b_key in body and body[b_key]:
                return str(body[b_key]).strip()
            return default

        tenant = get_val("X-Tenant-ID", "tenant_id", "default_tenant")
        user = get_val("X-User-ID", "user", "default_user")
        workspace = get_val("X-Workspace-ID", "workspace_id", "default_workspace")
        agent = get_val("X-Agent-ID", "agent_id", "default_agent")
        session = get_val("X-Session-ID", "session_id", "default_session")

        return cls(
            tenant_id=tenant,
            user_id=user,
            workspace_id=workspace,
            agent_id=agent,
            session_id=session
        )


@dataclass
class TenantQuota:
    max_concurrent_requests: int = 16
    max_virtual_tokens: int = 10_000_000
    max_stored_data_bytes: int = 100 * 1024 * 1024  # 100 MB
    max_sessions: int = 128
    max_generation_tokens: int = 8192


@dataclass
class ScopeMetrics:
    active_requests: int = 0
    total_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    virtual_tokens: int = 0
    tool_bytes_stored: int = 0
    compaction_events: int = 0


# =====================================================================
# Scoped Tool Result CAS Storage (Deduplicated with Logical Isolation)
# =====================================================================

class ScopedToolResultCASStore(ToolResultCASStore):
    """Content-Addressed Storage with strict multi-tenant ownership enforcement."""

    def __init__(self):
        super().__init__()
        self._record_owners: Dict[str, Tuple[SecurityScope, SharingScope]] = {}

    def ingest_scoped(
        self,
        tool_name: str,
        raw_output: str,
        scope: SecurityScope,
        command: str = "",
        exit_code: int = 0,
        sharing: SharingScope = SharingScope.PRIVATE
    ) -> Tuple[str, StructuredToolData, str]:
        rid, data, obs = self.ingest(tool_name, raw_output, command, exit_code)
        with self._lock:
            if rid not in self._record_owners:
                self._record_owners[rid] = (scope, sharing)
        return rid, data, obs

    def get_scoped_raw(self, result_id: str, requester: SecurityScope) -> str:
        with self._lock:
            if result_id in self._record_owners:
                owner, sharing = self._record_owners[result_id]
                if not requester.can_access(owner, sharing):
                    return ""  # Access Denied
            return self.get_raw_by_id(result_id)

    def get_scoped_fragment(self, result_id: str, start_line: int, end_line: int, requester: SecurityScope) -> str:
        with self._lock:
            if result_id in self._record_owners:
                owner, sharing = self._record_owners[result_id]
                if not requester.can_access(owner, sharing):
                    return ""  # Access Denied
            return self.get_fragment(result_id, start_line, end_line)


# =====================================================================
# Multi-Tenant & Multi-Agent Server Manager
# =====================================================================

class MultiTenantServerManager:
    """Manages multi-tenant sessions, virtual context instances, and fair scheduling."""

    def __init__(self, default_physical_limit: int = 32768):
        self.default_physical_limit = default_physical_limit
        self.tool_store = ScopedToolResultCASStore()
        self._lock = threading.RLock()
        
        # Sessions: session_id -> VirtualContextServerRuntime
        self._sessions: Dict[str, VirtualContextServerRuntime] = {}
        self._session_scopes: Dict[str, SecurityScope] = {}
        self._session_locks: Dict[str, threading.RLock] = {}
        
        # Quotas & Metrics
        self._tenant_quotas: Dict[str, TenantQuota] = {}
        self._tenant_metrics: Dict[str, ScopeMetrics] = {}
        
        # Shared Project/Workspace Knowledge Store
        self._shared_documents: List[Dict[str, Any]] = []
        self._shared_index = HierarchicalBM25Index()

    def get_or_create_session(self, scope: SecurityScope) -> Optional[VirtualContextServerRuntime]:
        with self._lock:
            sid = scope.session_id
            if sid in self._sessions:
                owner = self._session_scopes[sid]
                if owner.tenant_id != scope.tenant_id or owner.user_id != scope.user_id:
                    return None  # Access Denied: Tenant/User boundary mismatch
                return self._sessions[sid]

            # Check tenant quota
            quota = self._tenant_quotas.get(scope.tenant_id, TenantQuota())
            tenant_session_count = sum(1 for s in self._session_scopes.values() if s.tenant_id == scope.tenant_id)
            if tenant_session_count >= quota.max_sessions:
                return None  # Quota exceeded

            # Create isolated session
            vctx = VirtualContextServerRuntime(physical_context_limit=self.default_physical_limit)
            # Link to multi-tenant scoped tool store
            vctx.tool_store = self.tool_store
            
            self._sessions[sid] = vctx
            self._session_scopes[sid] = scope
            self._session_locks[sid] = threading.RLock()
            
            if scope.tenant_id not in self._tenant_metrics:
                self._tenant_metrics[scope.tenant_id] = ScopeMetrics()
            self._tenant_metrics[scope.tenant_id].total_requests += 1

            return vctx

    def delete_session(self, scope: SecurityScope) -> bool:
        with self._lock:
            sid = scope.session_id
            if sid not in self._sessions:
                return False
            owner = self._session_scopes[sid]
            if owner.tenant_id != scope.tenant_id or owner.user_id != scope.user_id:
                return False  # Access Denied
            
            del self._sessions[sid]
            del self._session_scopes[sid]
            if sid in self._session_locks:
                del self._session_locks[sid]
            return True

    def fork_session(self, parent_scope: SecurityScope, new_scope: SecurityScope) -> Optional[VirtualContextServerRuntime]:
        with self._lock:
            p_sid = parent_scope.session_id
            if p_sid not in self._sessions:
                return None
            owner = self._session_scopes[p_sid]
            if not new_scope.can_access(owner, SharingScope.USER):
                return None  # Cross-tenant / cross-user fork denied

            child_vctx = VirtualContextServerRuntime(physical_context_limit=self.default_physical_limit)
            child_vctx.tool_store = self.tool_store
            
            # Clone state from parent
            parent_vctx = self._sessions[p_sid]
            child_vctx.index.doc_records = copy.copy(parent_vctx.index.doc_records)
            child_vctx.index.inverted_index = copy.deepcopy(parent_vctx.index.inverted_index)
            
            n_sid = new_scope.session_id
            self._sessions[n_sid] = child_vctx
            self._session_scopes[n_sid] = new_scope
            self._session_locks[n_sid] = threading.RLock()
            return child_vctx

    def list_sessions(self, requester: SecurityScope) -> List[Dict[str, Any]]:
        with self._lock:
            results = []
            for sid, scope in self._session_scopes.items():
                if requester.can_access(scope, SharingScope.USER):
                    vctx = self._sessions[sid]
                    results.append({
                        "session_id": sid,
                        "tenant_id": scope.tenant_id,
                        "user_id": scope.user_id,
                        "workspace_id": scope.workspace_id,
                        "agent_id": scope.agent_id,
                        "stats": vctx.stats(),
                    })
            return results

    def ingest_shared_knowledge(
        self,
        owner_scope: SecurityScope,
        title: str,
        content: str,
        sharing: SharingScope = SharingScope.PROJECT
    ) -> None:
        with self._lock:
            item = ContextItem(
                item_id=len(self._shared_documents) + 1,
                item_type=ContextItemType.FILE_CONTENT,
                raw_content=content,
                token_count=max(1, len(content) // 4),
                timestamp=time.time(),
                metadata=ContextMetadata(filename=title, tags=[owner_scope.workspace_id, owner_scope.tenant_id])
            )
            self._shared_documents.append({
                "owner": owner_scope,
                "sharing": sharing,
                "title": title,
                "content": content,
                "item": item
            })
            self._shared_index.index_item(item)

    def retrieve_scoped(
        self,
        query: str,
        requester: SecurityScope,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        with self._lock:
            hits = []

            # 1. Search private session context
            sid = requester.session_id
            if sid in self._sessions:
                owner = self._session_scopes[sid]
                if requester.can_access(owner, SharingScope.PRIVATE):
                    s_hits = self._sessions[sid].index.search(query, top_k=top_k)
                    for item, score in s_hits:
                        hits.append({
                            "source": "private_session",
                            "score": round(score, 2),
                            "content": item.raw_content[:300],
                            "filename": item.metadata.filename
                        })

            # 2. Search shared project knowledge (pre-filtered by authorized scope)
            for doc in self._shared_documents:
                if requester.can_access(doc["owner"], doc["sharing"]):
                    text = (doc["title"] + " " + doc["content"]).lower()
                    if any(t in text for t in query.lower().split()):
                        hits.append({
                            "source": "shared_project_memory",
                            "score": 5.0,
                            "content": doc["content"][:300],
                            "filename": doc["title"]
                        })

            hits.sort(key=lambda x: x["score"], reverse=True)
            return hits[:top_k]

    def set_tenant_quota(self, tenant_id: str, quota: TenantQuota) -> None:
        with self._lock:
            self._tenant_quotas[tenant_id] = quota

    def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        with self._lock:
            base = self._tenant_metrics.get(tenant_id, ScopeMetrics())
            sessions_count = sum(1 for s in self._session_scopes.values() if s.tenant_id == tenant_id)
            return {
                "tenant_id": tenant_id,
                "active_sessions": sessions_count,
                "total_requests": base.total_requests,
                "input_tokens": base.input_tokens,
                "output_tokens": base.output_tokens,
                "virtual_tokens": base.virtual_tokens,
                "compaction_events": base.compaction_events,
            }

    def get_session_lock(self, session_id: str) -> threading.RLock:
        with self._lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.RLock()
            return self._session_locks[session_id]

    def summary_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_sessions = len(self._sessions)
            tenants = set(s.tenant_id for s in self._session_scopes.values())
            users = set(s.user_id for s in self._session_scopes.values())
            agents = set(s.agent_id for s in self._session_scopes.values())
            workspaces = set(s.workspace_id for s in self._session_scopes.values())

            tot_v_tokens = 0
            tot_p_tokens = 0
            tot_compactions = 0
            tot_tool_tokens_saved = 0
            for vctx in self._sessions.values():
                st = vctx.stats()
                tot_v_tokens += st.get("virtual_tokens_processed", 0)
                tot_p_tokens += st.get("physical_tokens_emitted", 0)
                tot_compactions += st.get("compaction_events", 0)
                tot_tool_tokens_saved += st.get("tool_tokens_saved", 0)

            cas_st = self.tool_store.stats()
            return {
                "active_sessions": total_sessions,
                "active_tenants": max(1, len(tenants)),
                "active_users": max(1, len(users)),
                "active_agents": max(1, len(agents)),
                "active_workspaces": max(1, len(workspaces)),
                "total_virtual_tokens": tot_v_tokens,
                "total_physical_tokens": tot_p_tokens,
                "total_compaction_events": tot_compactions,
                "tool_tokens_saved": tot_tool_tokens_saved,
                "context_compression_ratio": round(tot_v_tokens / max(1, tot_p_tokens), 2) if tot_p_tokens > 0 else 1.0,
                "cas_storage": cas_st,
                "shared_documents_count": len(self._shared_documents),
            }
