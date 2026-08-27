#ifndef XCALIBUR_POWER_FPGA_H
#define XCALIBUR_POWER_FPGA_H

// ---------------------------------------------------------------------------
// Board power sampling for Alveo cards.
//
// Unlike NVML on the GPU side, XRT exposes no cumulative energy counter for
// Alveo cards, only instantaneous power, so energy has to be integrated from
// samples. The card's XMC refreshes its sensors once per second (measured on a
// U280), which sets the accuracy floor: a run has to last tens of seconds for
// the integral to be meaningful, and each end of the window carries roughly one
// sample of uncertainty.
//
// Total board power is the sum of the three supply rails, which reproduces the
// figure xbutil reports to the last digit:  12V PEX + 12V AUX + 3V3 PEX.
//
// PowerMeter mirrors the NVML-backed type in power_nvidia.h call for call
// (open / measure_idle / start / finish / close + energy_j), so the host code
// around it is identical on both platforms even though what happens underneath
// is not.
// ---------------------------------------------------------------------------

#include <CL/cl.h>

#include <dirent.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "power_report.h"

// From cl_ext_xilinx.h. Defined here as a fallback so this header also compiles
// against an OpenCL install without the Xilinx extension headers.
#ifndef CL_DEVICE_PCIE_BDF
#define CL_DEVICE_PCIE_BDF 0x1120
#endif

static inline bool read_sysfs_long(const std::string& path, long* out) {
    std::ifstream f(path);
    long v = 0;
    if (!f || !(f >> v)) return false;
    *out = v;
    return true;
}

// The XMC node is named xmc.<instance>, so the instance number has to be found
// rather than hardcoded.
static inline std::string find_xmc_dir(const std::string& bdf) {
    std::string base = "/sys/bus/pci/devices/" + bdf + "/";
    DIR* d = opendir(base.c_str());
    if (!d) return "";
    std::string found;
    while (struct dirent* e = readdir(d)) {
        std::string name(e->d_name);
        if (name.rfind("xmc.", 0) == 0) {
            found = base + name + "/";
            break;
        }
    }
    closedir(d);
    return found;
}

// Rails are reported in millivolts and milliamps.
static inline bool read_board_power(const std::string& xmc, double* watts) {
    static const char* rails[3][2] = {
        {"xmc_12v_pex_vol", "xmc_12v_pex_curr"},
        {"xmc_12v_aux_vol", "xmc_12v_aux_curr"},
        {"xmc_3v3_pex_vol", "xmc_3v3_pex_curr"},
    };
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        long mv = 0, ma = 0;
        if (!read_sysfs_long(xmc + rails[i][0], &mv)) return false;
        if (!read_sysfs_long(xmc + rails[i][1], &ma)) return false;
        sum += (mv / 1000.0) * (ma / 1000.0);
    }
    *watts = sum;
    return true;
}

// Samples every 250 ms -- oversampling the 1 Hz sensor so each refresh is picked
// up promptly -- and accumulates energy trapezoidally. Best-effort, exactly like
// the NVML path on the GPU: any failure warns and reports no energy rather than
// taking the benchmark down with it.
struct PowerMeter {
    std::string xmc;
    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::thread th;
    double energy_j = 0.0;
    unsigned samples = 0;
    double last_w = 0.0;
    std::chrono::steady_clock::time_point last_t;
    bool active = false;

    bool open(cl_device_id device) {
        char bdf[64] = {0};
        if (clGetDeviceInfo(device, CL_DEVICE_PCIE_BDF, sizeof(bdf) - 1, bdf, nullptr) !=
            CL_SUCCESS) {
            std::cerr << "power: device BDF unavailable, energy not measured\n";
            return false;
        }
        xmc = find_xmc_dir(bdf);
        if (xmc.empty()) {
            std::cerr << "power: no XMC sysfs node under " << bdf << ", energy not measured\n";
            return false;
        }
        double w = 0.0;
        if (!read_board_power(xmc, &w)) {
            std::cerr << "power: cannot read the rails in " << xmc << ", energy not measured\n";
            return false;
        }
        active = true;
        return true;
    }

    // Baseline with the bitstream loaded and the queue empty. Runs synchronously
    // before the timed loop and leaves energy_j untouched. The 1 Hz XMC refresh is
    // why this window has to be seconds long rather than milliseconds.
    bool measure_idle(double seconds, double* watts) {
        if (!active) return false;
        double p_prev = 0.0;
        if (!read_board_power(xmc, &p_prev)) return false;
        auto t0 = std::chrono::steady_clock::now();
        auto t_prev = t0;
        double e = 0.0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            double p = 0.0;
            if (!read_board_power(xmc, &p)) return false;
            auto t = std::chrono::steady_clock::now();
            e += 0.5 * (p_prev + p) * std::chrono::duration<double>(t - t_prev).count();
            p_prev = p;
            t_prev = t;
            if (std::chrono::duration<double>(t - t0).count() >= seconds) break;
        }
        double span = std::chrono::duration<double>(t_prev - t0).count();
        if (span <= 0.0) return false;
        *watts = e / span;
        return true;
    }

    void start() {
        if (!active) return;
        if (!read_board_power(xmc, &last_w)) {
            failed = true;
            active = false;
            return;
        }
        last_t = std::chrono::steady_clock::now();
        th = std::thread([this] {
            while (!stop.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                accumulate();
            }
        });
    }

    void accumulate() {
        double w = 0.0;
        if (!read_board_power(xmc, &w)) {
            failed = true;
            return;
        }
        auto t = std::chrono::steady_clock::now();
        energy_j += 0.5 * (last_w + w) * std::chrono::duration<double>(t - last_t).count();
        samples++;
        last_w = w;
        last_t = t;
    }

    // Closes the window with one final sample so the tail between the last
    // periodic sample and the end of the loop is not dropped.
    bool finish() {
        if (!active) return false;
        stop = true;
        if (th.joinable()) th.join();
        accumulate();
        if (failed.load()) {
            std::cerr << "power: a rail read failed during the run, energy not measured\n";
            return false;
        }
        return true;
    }

    // Nothing to tear down: finish() already joined the sampling thread. Present
    // only so the host code around PowerMeter reads the same on both platforms,
    // where the NVML side really does have a library to shut down.
    void close() {
        if (th.joinable()) {
            stop = true;
            th.join();
        }
        active = false;
    }
};

#endif  // XCALIBUR_POWER_FPGA_H
