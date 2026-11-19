#ifndef XCALIBUR_POWER_REPORT_H
#define XCALIBUR_POWER_REPORT_H

// ---------------------------------------------------------------------------
// The per-run CSV summary every energy-instrumented app prints.
//
// One writer for all of them: xcalibur/scripts/motivation-perf-power.py parses
// this by matching an `app_name,...` header line against the data line that
// follows, and asserts that app_name/in_size/out_size/reps_warmup/reps/
// idle_window_s are identical across every run of an app. A field that drifts
// between two apps' hand-written ostream chains would surface as a parse error
// far from its cause, so the format lives here and nowhere else.
//
// Unavailable measurements are written as empty fields rather than zero, so a
// missing measurement cannot be mistaken for a device that drew no power. The
// row always has 15 fields either way.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <iostream>

static inline void print_power_csv(const char* app_name, size_t in_size, size_t out_size,
                                   int n_warmup, int n_reps, double time_xpu_s,
                                   double time_data_to_xpu_s, double time_kernel_s,
                                   double time_data_to_host_s, double time_loop_s,
                                   bool have_energy, double energy_j, bool have_idle,
                                   double idle_w, double idle_window_s) {
    std::cout << "app_name,in_size,out_size,reps_warmup,reps,time_xpu,time_data_to_xpu,time_kernel,"
                 "time_data_to_host,time_loop,energy_j,avg_power_w,idle_w,idle_window_s,"
                 "net_energy_j\n"
              << app_name << ","
              << in_size << ","
              << out_size << ","
              << n_warmup << ","
              << n_reps << ","
              << time_xpu_s << ","
              << time_data_to_xpu_s << ","
              << time_kernel_s << ","
              << time_data_to_host_s << ","
              << time_loop_s << ",";
    if (have_energy) {
        std::cout << energy_j << "," << (time_loop_s > 0.0 ? energy_j / time_loop_s : 0.0);
    } else {
        std::cout << ",";
    }
    std::cout << ",";
    if (have_idle) {
        std::cout << idle_w << "," << idle_window_s;
    } else {
        std::cout << ",";
    }
    std::cout << ",";
    // Energy above the idle baseline: what running the workload actually cost.
    if (have_energy && have_idle) {
        std::cout << energy_j - idle_w * time_loop_s;
    }
    std::cout << "\n";
}

#endif  // XCALIBUR_POWER_REPORT_H
