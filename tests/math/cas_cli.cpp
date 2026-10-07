// tests/math/cas_cli.cpp - evaluates one expression per input line with the native CAS (used by the differential tests)
//   input line:  <expression>        output line: <result> | ERROR: <message> | UNSUPPORTED: <message> | LIMIT: <message>
#include "strata/math/cas/engine.hpp"

#include <iostream>

using namespace strata::math::cas;

int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) { std::cout << "\n"; continue; }
        try {
            Engine en;
            std::cout << to_string(en.eval(parse(line))) << "\n";
        } catch (const CasUnsupported& e) { std::cout << "UNSUPPORTED: " << e.what() << "\n"; }
        catch (const CasLimitError& e) { std::cout << "LIMIT: " << e.what() << "\n"; }
        catch (const CasError& e) { std::cout << "ERROR: " << e.what() << "\n"; }
        catch (const std::exception& e) { std::cout << "ERROR: " << e.what() << "\n"; }
        std::cout.flush();
    }
    return 0;
}
