// include/strata/tools/tool_parser.hpp - Tool Result Parser & State Generator
//
// Automatically extracts structured state (compiler errors, test failures, git changes,
// search matches) from raw textual output without forcing the LLM to parse raw logs.
#pragma once

#include "strata/tools/tool_types.hpp"

#include <memory>
#include <string>

namespace strata::tools {

class ToolResultParser {
public:
    virtual ~ToolResultParser() = default;

    virtual bool can_parse(const std::string& tool_name, const std::string& raw_output) const = 0;
    virtual StructuredToolData parse(const std::string& tool_name, const std::string& raw_output,
                                     int exit_code) const = 0;
};

class CompilerOutputParser : public ToolResultParser {
public:
    bool can_parse(const std::string& tool_name, const std::string& raw_output) const override;
    StructuredToolData parse(const std::string& tool_name, const std::string& raw_output,
                             int exit_code) const override;
};

class TestOutputParser : public ToolResultParser {
public:
    bool can_parse(const std::string& tool_name, const std::string& raw_output) const override;
    StructuredToolData parse(const std::string& tool_name, const std::string& raw_output,
                             int exit_code) const override;
};

class SearchOutputParser : public ToolResultParser {
public:
    bool can_parse(const std::string& tool_name, const std::string& raw_output) const override;
    StructuredToolData parse(const std::string& tool_name, const std::string& raw_output,
                             int exit_code) const override;
};

class CompositeToolParser : public ToolResultParser {
public:
    CompositeToolParser();

    bool can_parse(const std::string& tool_name, const std::string& raw_output) const override;
    StructuredToolData parse(const std::string& tool_name, const std::string& raw_output,
                             int exit_code) const override;

    void register_parser(std::shared_ptr<ToolResultParser> parser);

private:
    std::vector<std::shared_ptr<ToolResultParser>> parsers_;
    StructuredToolData parse_generic(const std::string& raw_output, int exit_code) const;
};

} // namespace strata::tools
