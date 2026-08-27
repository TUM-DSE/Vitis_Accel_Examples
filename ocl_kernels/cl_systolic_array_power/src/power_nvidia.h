#ifndef XCALIBUR_POWER_NVIDIA_H
#define XCALIBUR_POWER_NVIDIA_H

// ---------------------------------------------------------------------------
// Board energy for NVIDIA GPUs, via NVML's cumulative counter.
//
// nvmlDeviceGetTotalEnergyConsumption reports board-wide millijoules since the
// driver was last reloaded. It is monotonic, so a window's energy is just the
// difference of two reads -- no sampling thread and no integration error, which
// is why this side is exact where the Alveo side in power_fpga.h is not. Needs
// Volta or newer; older GPUs answer NVML_ERROR_NOT_SUPPORTED.
//
// PowerMeter mirrors the XMC-backed type in power_fpga.h call for call
// (open / measure_idle / start / finish / close + energy_j), so the host code
// around it is identical on both platforms even though what happens underneath
// is not.
// ---------------------------------------------------------------------------

#include <CL/cl.h>
#include <nvml.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

#include "power_report.h"

// From cl_nv_device_attribute_query, so NVML can be pointed at the very board
// OpenCL selected instead of assuming there is only one GPU in the machine.
#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008
#endif
#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009
#endif

struct PowerMeter {
    nvmlDevice_t dev{};
    bool nvml_up = false;
    bool active = false;
    unsigned long long start_mj = 0;
    double energy_j = 0.0;

    bool read_mj(unsigned long long* mj) const {
        return nvmlDeviceGetTotalEnergyConsumption(dev, mj) == NVML_SUCCESS;
    }

    // Binds NVML to the OpenCL device, matching on PCI address and falling back
    // to NVML index 0 if the driver does not expose the extension. Energy
    // measurement is best-effort: every failure path warns and leaves the
    // benchmark itself running, it just reports no energy.
    bool open(cl_device_id device) {
        nvmlReturn_t r = nvmlInit();
        if (r != NVML_SUCCESS) {
            std::cerr << "NVML: init failed (" << nvmlErrorString(r) << "), energy not measured\n";
            return false;
        }
        nvml_up = true;

        cl_uint bus = 0, slot = 0;
        bool bound = false;
        if (clGetDeviceInfo(device, CL_DEVICE_PCI_BUS_ID_NV, sizeof(bus), &bus, nullptr) ==
                CL_SUCCESS &&
            clGetDeviceInfo(device, CL_DEVICE_PCI_SLOT_ID_NV, sizeof(slot), &slot, nullptr) ==
                CL_SUCCESS) {
            char pci[32];
            snprintf(pci, sizeof(pci), "0000:%02x:%02x.0", bus, slot);
            r = nvmlDeviceGetHandleByPciBusId(pci, &dev);
            if (r == NVML_SUCCESS) {
                bound = true;
            } else {
                std::cerr << "NVML: no device at " << pci << " (" << nvmlErrorString(r)
                          << "), falling back to NVML index 0\n";
            }
        }
        if (!bound) {
            r = nvmlDeviceGetHandleByIndex(0, &dev);
            if (r != NVML_SUCCESS) {
                std::cerr << "NVML: no device (" << nvmlErrorString(r)
                          << "), energy not measured\n";
                return false;
            }
        }

        unsigned long long probe = 0;
        if (!read_mj(&probe)) {
            std::cerr << "NVML: total energy counter unsupported on this GPU"
                         " (needs Volta or newer), energy not measured\n";
            return false;
        }
        active = true;
        return true;
    }

    // Baseline in the same process state as the run that follows: context and
    // program built, buffers allocated, nothing enqueued. Differencing the
    // counter gives an exact mean over the window rather than a noisy
    // instantaneous sample.
    //
    // This assumes the GPU clocks are pinned (nvidia-smi -pm 1 / -lgc). Without
    // that the P-state here need not be the one that applies during the loop,
    // and the baseline would not be the right thing to subtract.
    bool measure_idle(double seconds, double* watts) {
        if (!active) return false;
        unsigned long long e0 = 0, e1 = 0;
        if (!read_mj(&e0)) return false;
        auto t0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        auto t1 = std::chrono::steady_clock::now();
        double span = std::chrono::duration<double>(t1 - t0).count();
        if (!read_mj(&e1) || span <= 0.0) return false;
        *watts = (e1 - e0) / 1000.0 / span;
        return true;
    }

    void start() {
        if (!active) return;
        if (!read_mj(&start_mj)) active = false;
    }

    bool finish() {
        if (!active) return false;
        unsigned long long end_mj = 0;
        if (!read_mj(&end_mj)) {
            std::cerr << "NVML: reading the energy counter after the loop failed,"
                         " energy not measured\n";
            active = false;
            return false;
        }
        energy_j = (end_mj - start_mj) / 1000.0;
        return true;
    }

    // Safe to call whether or not open() succeeded.
    void close() {
        if (nvml_up) nvmlShutdown();
        nvml_up = false;
        active = false;
    }
};

#endif  // XCALIBUR_POWER_NVIDIA_H
