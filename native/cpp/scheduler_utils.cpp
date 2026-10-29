#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <algorithm>

namespace py = pybind11;

struct RequestCandidate {
    std::string request_id;
    double arrival_time;
    int num_tokens;
    int priority;
};

// Fast C++ request prioritization with Python GIL released
std::vector<std::string> prioritize_requests_cpp(
    const std::vector<std::string>& request_ids,
    const std::vector<double>& arrival_times,
    const std::vector<int>& token_counts,
    const std::string& policy
) {
    if (request_ids.empty()) {
        return {};
    }

    size_t n = request_ids.size();
    std::vector<RequestCandidate> candidates(n);

    // Release the Python Global Interpreter Lock (GIL) during C++ computation
    {
        py::gil_scoped_release release;

        for (size_t i = 0; i < n; ++i) {
            candidates[i] = {
                request_ids[i],
                arrival_times[i],
                token_counts[i],
                static_cast<int>(i)
            };
        }

        if (policy == "sjf") { // Shortest Job First
            std::sort(candidates.begin(), candidates.end(), [](const RequestCandidate& a, const RequestCandidate& b) {
                if (a.num_tokens != b.num_tokens) {
                    return a.num_tokens < b.num_tokens;
                }
                return a.arrival_time < b.arrival_time;
            });
        } else { // Default: FCFS (First-Come, First-Served)
            std::sort(candidates.begin(), candidates.end(), [](const RequestCandidate& a, const RequestCandidate& b) {
                return a.arrival_time < b.arrival_time;
            });
        }
    }

    std::vector<std::string> sorted_ids(n);
    for (size_t i = 0; i < n; ++i) {
        sorted_ids[i] = candidates[i].request_id;
    }
    return sorted_ids;
}
