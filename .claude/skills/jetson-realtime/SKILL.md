---
name: jetson-realtime
description: Real-time inference on NVIDIA Jetson Orin Nano Super — latency budgets, the 8 GB shared-memory constraint, TensorRT conversion, camera pipelines, thermal behaviour, and how to profile before optimising. Use this whenever writing or reviewing code that runs on the kiosk device, choosing or sizing a model for the edge, debugging dropped frames or latency, or planning what can realistically run on-device versus in the cloud.
---

# Real-time inference on Jetson Orin Nano Super

The target device is the [Orin Nano Super](https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/): 67 sparse TOPS, 1024 CUDA cores, 32 tensor cores, 6-core Cortex-A78AE at 1.7 GHz, **8 GB LPDDR5 at 102 GB/s**.

The TOPS number is not the constraint. **Memory is**, and so is the fact that it is
shared.

## The 8 GB is shared, and that changes the design

CPU and GPU draw from the same 8 GB. There is no separate VRAM to spill into. Subtract
the OS and desktop (~1.5 GB on a kiosk with a display) and roughly **6–6.5 GB** is
left for everything: perception, recognition, any language model, speech synthesis,
video buffers, and the application.

Budget it explicitly, in writing, before choosing models. A design that assumes a 7B
language model at 4-bit (~4.5 GB plus KV cache) has spent the entire budget on one
component and will start swapping — which on this class of device means multi-second
stalls, not graceful degradation.

Practical consequences:

- Size language models at the **3B class quantised** if they must run locally, or route
  to a server and keep a small local model purely as the offline fallback. Decide this
  early; it shapes the whole product, not just a config value.
- Load models **once at startup**, never per request. Loading is seconds, not
  milliseconds, and a reload mid-session reads to the user as a crash.
- Watch for **memory growth**, not just peak. A kiosk runs for weeks. Frame buffers
  that accumulate, per-session caches that never evict, and CUDA fragmentation all
  show up on day nine, not in a demo.
- Set an explicit **swap policy**. Silent swapping is the most common cause of "it was
  fast yesterday" on Jetson.

## Frame rate: capture fast, infer slower

A 60 fps camera does not mean 60 inferences per second, and treating it that way
wastes the entire compute budget for no accuracy gain.

Sign recognition needs enough temporal resolution to catch fast handshape changes;
past roughly 30 fps the extra frames are largely redundant, because human signing does
not carry information at that rate.

The sound pattern:

- **Capture at 60 fps** — reduces motion blur, gives the preview a responsive feel, and
  lets you pick the freshest frame rather than waiting for the next one.
- **Run perception at 30 fps** on every second frame.
- **Run classification on a sliding window**, several times per second — not per frame.
- **Always drop, never queue.** If perception is not ready, discard the frame. A queue
  that grows converts a throughput problem into an ever-increasing latency problem, and
  the user sees the system responding to what they signed several seconds ago. Latency
  is far more damaging than a dropped frame here.

Keep capture, inference, and UI on separate threads with a single-slot buffer between
them — newest frame wins, older frame discarded.

## Latency budget

Write the budget down and measure against it. For a kiosk, the number that matters is
**end of sign → text on screen**, and it should stay under ~300 ms for the interaction
to feel like a conversation rather than a form submission.

Allocate, then verify each stage independently:

```
camera capture + colour convert     ~5 ms
landmark extraction                 ~20 ms   ← usually the largest, optimise here first
feature assembly                     ~2 ms
classification                       ~5 ms
UI render                           ~16 ms   (one frame at 60 Hz)
```

Speech synthesis and any language-model answer sit **outside** this budget — they
happen after the text appears. Show text the instant it is ready; never make the user
wait for audio they cannot hear anyway.

## TensorRT

Convert models to TensorRT for deployment; expect roughly 2–4× over plain PyTorch on
this hardware, mostly from kernel fusion and FP16.

Things that cost people days:

- **Engines are not portable.** A TensorRT engine is tied to the GPU architecture,
  TensorRT version, and JetPack version. Build on the target device or in a container
  matching it exactly, and treat the engine as a build artefact — never commit it as
  the source of truth. Keep the ONNX and rebuild.
- **Fix input shapes** where possible. Dynamic shapes cost performance and complicate
  the engine.
- **FP16 is usually free accuracy-wise** for these models; INT8 needs calibration data
  and must be re-validated on the frozen benchmark, because quantisation can degrade
  specific classes badly while leaving the average intact.
- **Verify numerically after conversion.** Run the same inputs through PyTorch and the
  engine and compare outputs. A silently wrong engine is entirely possible.

## Thermal and power

Sustained load throttles. A kiosk is sustained load, all day.

- Set the power mode explicitly (`nvpmodel`) and pin clocks (`jetson_clocks`) as part
  of service startup, so behaviour does not depend on what someone last ran by hand.
- **Benchmark hot, not cold.** Numbers from the first 30 seconds after boot are
  fiction. Run for 20+ minutes and measure the steady state.
- Log temperature and throttle state to telemetry. A kiosk slowing down in August is
  otherwise indistinguishable from a software regression.
- Verify the enclosure has real airflow. A sealed kiosk cabinet throttles a board that
  runs fine on a desk.

## Profile before optimising

`tegrastats` alongside application timers is enough to answer the only question worth
asking first: is this stage GPU-bound, CPU-bound, memory-bound, or waiting on I/O? The
common surprises on this board are that image colour conversion and tensor copies
between CPU and GPU cost far more than expected, and that Python-side preprocessing
dominates a pipeline everyone assumed was GPU-bound.

Measure per stage, with timers that stay in the code and feed telemetry. A kiosk in the
field cannot be attached to a profiler, and the slow stage in a mall is not always the
slow stage on a desk.
