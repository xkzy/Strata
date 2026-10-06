// src/tools/tool_parser.cpp - Tool Result Parser Implementation
#include "strata/tools/tool_parser.hpp"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace strata::tools {

namespace {

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return lines;
}

} // anonymous namespace

// ------------------- Compiler Output Parser -------------------
bool CompilerOutputParser::can_parse(const std::string& tool_name, const std::string& raw_output) const {
    if (tool_name == "ninja" || tool_name == "make" || tool_name == "gcc" || tool_name == "clang" ||
        tool_name == "cl" || tool_name == "nvcc") {
        return true;
    }
    return (raw_output.find("error:") != std::string::npos ||
            raw_output.find("warning:") != std::string::npos ||
            raw_output.find("FAILED:") != std::string::npos);
}

StructuredToolData CompilerOutputParser::parse(const std::string& /*tool_name*/,
                                              const std::string& raw_output,
                                              int exit_code) const {
    StructuredToolData data;
    data.is_build = true;
    data.exit_code = exit_code;
    data.build_success = (exit_code == 0);

    auto lines = split_lines(raw_output);
    data.total_lines = lines.size();

    std::regex diag_regex(R"((.+?):(\d+):(\d+):\s*(error|warning|note):\s*(.+))");
    std::smatch match;

    for (const auto& line : lines) {
        if (std::regex_search(line, match, diag_regex)) {
            CompilerDiagnostic diag;
            diag.file = match[1].str();
            diag.line = std::stoi(match[2].str());
            diag.column = std::stoi(match[3].str());
            diag.severity = match[4].str();
            diag.message = match[5].str();

            if (diag.severity == "error") {
                data.error_count++;
                data.build_success = false;
            } else if (diag.severity == "warning") {
                data.warning_count++;
            }
            data.diagnostics.push_back(diag);
        } else if (line.find("error:") != std::string::npos || line.find("FAILED:") != std::string::npos) {
            data.error_count++;
            data.build_success = false;
        } else if (line.find("warning:") != std::string::npos) {
            data.warning_count++;
        }
    }

    return data;
}

// ------------------- Test Output Parser -------------------
bool TestOutputParser::can_parse(const std::string& tool_name, const std::string& raw_output) const {
    if (tool_name == "pytest" || tool_name == "ctest" || tool_name == "test") return true;
    return (raw_output.find("test session starts") != std::string::npos ||
            raw_output.find("Test #") != std::string::npos ||
            raw_output.find("passed") != std::string::npos ||
            raw_output.find("FAILED") != std::string::npos);
}

StructuredToolData TestOutputParser::parse(const std::string& /*tool_name*/,
                                          const std::string& raw_output,
                                          int exit_code) const {
    StructuredToolData data;
    data.is_test = true;
    data.exit_code = exit_code;

    auto lines = split_lines(raw_output);
    data.total_lines = lines.size();

    std::regex pytest_summary_regex(R"((\d+)\s+passed(?:,\s+(\d+)\s+failed)?)");
    std::regex failed_test_regex(R"(FAILED\s+([^\s:]+))");
    std::smatch match;

    for (const auto& line : lines) {
        if (std::regex_search(line, match, failed_test_regex)) {
            TestFailure f;
            f.test_name = match[1].str();
            f.failure_message = line;
            data.failures.push_back(f);
            data.tests_failed++;
        } else if (line.find("Passed") != std::string::npos || line.find("PASSED") != std::string::npos) {
            data.tests_passed++;
        }
    }

    if (data.tests_failed > 0 || exit_code != 0) {
        data.build_success = false;
    }

    return data;
}

// ------------------- Search Output Parser -------------------
bool SearchOutputParser::can_parse(const std::string& tool_name, const std::string& raw_output) const {
    if (tool_name == "grep" || tool_name == "find" || tool_name == "rg" || tool_name == "ag") return true;
    return (raw_output.find("matches") != std::string::npos || raw_output.find("Searching") != std::string::npos);
}

StructuredToolData SearchOutputParser::parse(const std::string& /*tool_name*/,
                                            const std::string& raw_output,
                                            int exit_code) const {
    StructuredToolData data;
    data.is_search = true;
    data.exit_code = exit_code;

    auto lines = split_lines(raw_output);
    data.total_lines = lines.size();
    data.total_matches = lines.size();

    std::unordered_set<std::string> unique_files;
    for (size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos && colon_pos > 0 && colon_pos < 100) {
            unique_files.insert(line.substr(0, colon_pos));
        }
        if (data.key_excerpts.size() < 5 && !line.empty()) {
            data.key_excerpts.push_back(line);
        }
    }

    for (const auto& f : unique_files) {
        data.matched_files.push_back(f);
    }

    return data;
}

// ------------------- Composite Tool Parser -------------------
CompositeToolParser::CompositeToolParser() {
    register_parser(std::make_shared<CompilerOutputParser>());
    register_parser(std::make_shared<TestOutputParser>());
    register_parser(std::make_shared<SearchOutputParser>());
}

void CompositeToolParser::register_parser(std::shared_ptr<ToolResultParser> parser) {
    if (parser) {
        parsers_.push_back(std::move(parser));
    }
}

bool CompositeToolParser::can_parse(const std::string&, const std::string&) const {
    return true; // Fallback to generic parsing
}

StructuredToolData CompositeToolParser::parse_generic(const std::string& raw_output, int exit_code) const {
    StructuredToolData data;
    data.exit_code = exit_code;
    auto lines = split_lines(raw_output);
    data.total_lines = lines.size();
    data.build_success = (exit_code == 0);

    for (size_t i = 0; i < std::min<size_t>(lines.size(), 3); ++i) {
        if (!lines[i].empty()) {
            data.key_excerpts.push_back(lines[i]);
        }
    }
    return data;
}

StructuredToolData CompositeToolParser::parse(const std::string& tool_name,
                                             const std::string& raw_output,
                                             int exit_code) const {
    for (const auto& p : parsers_) {
        if (p->can_parse(tool_name, raw_output)) {
            return p->parse(tool_name, raw_output, exit_code);
        }
    }
    return parse_generic(raw_output, exit_code);
}

} // namespace strata::tools
