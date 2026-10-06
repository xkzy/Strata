// src/guard/execution_graph.cpp - Execution Graph Implementation
#include "strata/guard/execution_graph.hpp"

#include <algorithm>
#include <functional>

namespace strata::guard {

ExecutionGraph::ExecutionGraph() = default;
ExecutionGraph::~ExecutionGraph() = default;

bool ExecutionGraph::add_node(const ActionRecord& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (record.action_id.empty()) return false;

    GraphNode node;
    node.record = record;
    node.parent_id = record.parent_id;

    if (!record.parent_id.empty()) {
        auto it = nodes_.find(record.parent_id);
        if (it != nodes_.end()) {
            it->second.children_ids.push_back(record.action_id);
            node.depth = it->second.depth + 1;

            // Track agent-to-agent delegation
            if (!it->second.record.agent_id.empty() && !record.agent_id.empty() &&
                it->second.record.agent_id != record.agent_id) {
                agent_delegation_edges_[it->second.record.agent_id].push_back(record.agent_id);
            }
        } else {
            node.depth = 1;
        }
    } else {
        node.depth = 0;
    }

    nodes_[record.action_id] = std::move(node);
    return true;
}

bool ExecutionGraph::add_edge(const std::string& parent_id, const std::string& child_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto p_it = nodes_.find(parent_id);
    auto c_it = nodes_.find(child_id);
    if (p_it == nodes_.end() || c_it == nodes_.end()) return false;

    p_it->second.children_ids.push_back(child_id);
    c_it->second.parent_id = parent_id;
    c_it->second.depth = p_it->second.depth + 1;

    if (!p_it->second.record.agent_id.empty() && !c_it->second.record.agent_id.empty() &&
        p_it->second.record.agent_id != c_it->second.record.agent_id) {
        agent_delegation_edges_[p_it->second.record.agent_id].push_back(c_it->second.record.agent_id);
    }
    return true;
}

const GraphNode* ExecutionGraph::get_node(const std::string& action_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(action_id);
    return it != nodes_.end() ? &it->second : nullptr;
}

bool ExecutionGraph::detect_cycle(const std::string& action_id, std::vector<std::string>& cycle_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (nodes_.find(action_id) == nodes_.end()) return false;

    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> recursion_stack;
    std::vector<std::string> path;

    std::function<bool(const std::string&)> dfs = [&](const std::string& curr) -> bool {
        visited.insert(curr);
        recursion_stack.insert(curr);
        path.push_back(curr);

        auto it = nodes_.find(curr);
        if (it != nodes_.end()) {
            for (const auto& child : it->second.children_ids) {
                if (recursion_stack.find(child) != recursion_stack.end()) {
                    // Cycle detected
                    path.push_back(child);
                    cycle_path = path;
                    return true;
                }
                if (visited.find(child) == visited.end()) {
                    if (dfs(child)) return true;
                }
            }
        }

        path.pop_back();
        recursion_stack.erase(curr);
        return false;
    };

    return dfs(action_id);
}

bool ExecutionGraph::detect_cross_agent_cycle(const std::string& current_agent_id,
                                             std::vector<std::string>& cycle_agents) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (current_agent_id.empty() || agent_delegation_edges_.find(current_agent_id) == agent_delegation_edges_.end()) {
        return false;
    }

    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> in_stack;
    std::vector<std::string> path;

    std::function<bool(const std::string&)> dfs = [&](const std::string& agent) -> bool {
        visited.insert(agent);
        in_stack.insert(agent);
        path.push_back(agent);

        auto it = agent_delegation_edges_.find(agent);
        if (it != agent_delegation_edges_.end()) {
            for (const auto& next_agent : it->second) {
                if (in_stack.find(next_agent) != in_stack.end()) {
                    path.push_back(next_agent);
                    cycle_agents = path;
                    return true;
                }
                if (visited.find(next_agent) == visited.end()) {
                    if (dfs(next_agent)) return true;
                }
            }
        }

        path.pop_back();
        in_stack.erase(agent);
        return false;
    };

    return dfs(current_agent_id);
}

int ExecutionGraph::calculate_recursion_depth(const std::string& action_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(action_id);
    if (it == nodes_.end()) return 0;
    return it->second.depth;
}

int ExecutionGraph::calculate_fan_out(const std::string& parent_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(parent_id);
    if (it == nodes_.end()) return 0;
    return static_cast<int>(it->second.children_ids.size());
}

void ExecutionGraph::prune_older_than(double cutoff_sec) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> to_remove;
    for (const auto& kv : nodes_) {
        if (kv.second.record.timestamp_sec < cutoff_sec) {
            to_remove.push_back(kv.first);
        }
    }
    for (const auto& id : to_remove) {
        nodes_.erase(id);
    }
}

void ExecutionGraph::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    nodes_.clear();
    agent_delegation_edges_.clear();
}

size_t ExecutionGraph::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return nodes_.size();
}

} // namespace strata::guard
