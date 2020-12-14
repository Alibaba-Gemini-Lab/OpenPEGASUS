#include "pegasus/rotgroup_helper.h"

RotGroupHelper RotGroupHelper::instance_; // singleton
std::unordered_map<uint64_t, std::vector<size_t>> RotGroupHelper::rotGroups_;

template <typename T>
static inline bool islog2(T n) { return 0 == (n & (n - 1)); }

std::vector<size_t> RotGroupHelper::get(int degree, int nslots) {
    if (degree < 0 || nslots < 1 || !islog2(degree) || (degree % nslots != 0)) {
        return std::vector<size_t>();
    }

    instance_.guard_.lock_shared(); // R-lock
    uint64_t key = (static_cast<uint64_t>(degree) << 32) | static_cast<uint32_t>(nslots);
    auto val = instance_.rotGroups_.find(key);
    if (val != instance_.rotGroups_.end()) {
        instance_.guard_.unlock_shared(); // release R-lock
        return val->second;
    }
        
    instance_.guard_.unlock_shared(); // release the R-lock
    instance_.guard_.lock(); // apply the W-lock
    val = instance_.rotGroups_.find(key); 
    // double-check, the table might be updated when waiting for the W-lock
    if (val != instance_.rotGroups_.end()) {
        instance_.guard_.unlock();
        return val->second;
    }

    std::vector<size_t> rotGroup(nslots);
    size_t m = degree << 1;
    size_t genPows = 1;
    for (size_t i = 0; i < nslots; ++i) {
        rotGroup[i] = genPows;
        genPows *= gen;
        genPows &= (m - 1);
    }

    instance_.rotGroups_.insert({key, rotGroup});
    instance_.guard_.unlock();
    return rotGroup;
}
