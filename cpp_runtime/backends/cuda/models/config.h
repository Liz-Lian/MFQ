#pragma once

#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::cuda::config_json {

inline std::int64_t integer(
        const std::string& source,
        const std::string& key,
        std::int64_t fallback = -1) {
    const std::regex pattern(
        "\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (std::regex_search(source, match, pattern)) {
        return std::stoll(match[1].str());
    }
    if (fallback >= 0) return fallback;
    throw std::runtime_error("missing config int: " + key);
}

inline double number(
        const std::string& source,
        const std::string& key,
        double fallback) {
    const std::regex pattern(
        "\"" + key +
        "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
    std::smatch match;
    return std::regex_search(source, match, pattern)
        ? std::stod(match[1].str())
        : fallback;
}

inline bool boolean(
        const std::string& source,
        const std::string& key,
        bool fallback) {
    const std::regex pattern(
        "\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    return std::regex_search(source, match, pattern)
        ? match[1].str() == "true"
        : fallback;
}

inline std::string string(
        const std::string& source,
        const std::string& key,
        const std::string& fallback = {}) {
    const std::regex pattern(
        "\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    return std::regex_search(source, match, pattern)
        ? match[1].str()
        : fallback;
}

inline double object_number(
        const std::string& source,
        const std::string& object,
        const std::string& key,
        double fallback) {
    const std::regex pattern(
        "\"" + object + "\"\\s*:\\s*\\{[^}]*\"" + key +
        "\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)");
    std::smatch match;
    return std::regex_search(source, match, pattern)
        ? std::stod(match[1].str())
        : fallback;
}

inline std::vector<std::string> layer_types(
        const std::string& source,
        std::int64_t count) {
    std::vector<std::string> result;
    const std::regex pattern("\"layer_types\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (std::regex_search(source, match, pattern)) {
        const auto body = match[1].str();
        const std::regex item("\"([^\"]+)\"");
        for (auto it = std::sregex_iterator(body.begin(), body.end(), item);
             it != std::sregex_iterator(); ++it) {
            result.push_back((*it)[1].str());
        }
    }
    if (result.empty()) {
        result.assign(static_cast<std::size_t>(count), "full_attention");
    }
    return result;
}

inline std::vector<std::string> string_array(
        const std::string& source,
        const std::string& key) {
    std::vector<std::string> result;
    const std::regex pattern(
        "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(source, match, pattern)) return result;
    const auto body = match[1].str();
    const std::regex item("\"([^\"]+)\"");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item);
         it != std::sregex_iterator(); ++it) {
        result.push_back((*it)[1].str());
    }
    return result;
}

inline std::vector<std::int64_t> integer_array(
        const std::string& source,
        const std::string& key) {
    std::vector<std::int64_t> result;
    const std::regex pattern(
        "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(source, match, pattern)) return result;
    const auto body = match[1].str();
    const std::regex item("-?\\d+");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), item);
         it != std::sregex_iterator(); ++it) {
        result.push_back(std::stoll((*it)[0].str()));
    }
    return result;
}

} // namespace mfq::cuda::config_json
