// include/strata/guard/execution_graph.hpp - Execution Graph & Lineage Cycle Detection
#pragma once

#include "strata/guard/anti_loop_types.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strata::guard {

struct GraphNode {
    ActionRecord record;
    std::vector<std::string> children_ids;
    std::string parent_id;
    int depth = 0;
};

class ExecutionGraph {
public:
    ExecutionGraph();
    ~ExecutionGraph();

    // Register a new action node in the execution graph
    bool add_node(const ActionRecord& record);

    // Explicitly link parent and child operations
    bool add_edge(const std::string& parent_id, const std::string& child_id);

    // Retrieve a node by action ID
    const GraphNode* get_node(const std::string& action_id) const;

    // Detect if the node is part of a causal cycle in the execution DAG
    bool detect_cycle(const std::string& action_id, std::vector<std::string>& cycle_path) const;

    // Detect if cross-agent delegation has formed a closed loop (Agent A -> B -> A)
    bool detect_cross_agent_cycle(const std::string& current_agent_id,
                                  std::vector<std::string>& cycle_agents) const;

    // Calculate the depth of recursion/ancestry for an action
    int calculate_recursion_depth(const std::string& action_id) const;

    // Count the fan-out (number of direct child operations) for a parent
    int calculate_fan_out(const std::string& parent_id) const;

    // Clean up old nodes past retention window to keep memory overhead near zero
    void prune_older_than(double cutoff_sec);

    // Clear the execution graph
    void clear();

    // Node count
    size_t size() const;

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, GraphNode> nodes_;
    std::unordered_map<std::string, std::vector<std::string>> agent_delegation_edges_;
};

} // namespace strata::guard
