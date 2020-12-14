#pragma once
#include <unordered_map>
#include <vector>
#include "contrib/safe_ptr.h"

class RotGroupHelper {
public:
    static constexpr size_t gen = 5;
    /// Empty vector would be return if the inputs are invalid.
    static std::vector<size_t> get(int degree, int nslots);

private:
    RotGroupHelper() {}
    ~RotGroupHelper() {}
    RotGroupHelper(const RotGroupHelper &oth) = delete;
    RotGroupHelper(RotGroupHelper &&oth)      = delete;
    RotGroupHelper &operator=(const RotGroupHelper &oth) = delete;

    static RotGroupHelper instance_;  // singleton

    sf::contention_free_shared_mutex<> guard_;
    static std::unordered_map<uint64_t, std::vector<size_t>> rotGroups_;
};
