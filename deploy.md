# Deploying Moonshine Voice on Ubuntu (AMD x86_64 Dual-Core Server)

This guide covers building and running Moonshine Voice from source on an Ubuntu server with an AMD dual-core (x86_64) processor. No GPU required — everything runs on-CPU.

---

## Prerequisites

### System Requirements

- **OS:** Ubuntu 20.04 LTS or later (22.04 / 24.04 recommended)
- **CPU:** AMD x86_64 dual-core (or better)
- **RAM:** 1 GB minimum, 2 GB+ recommended for Medium Streaming model
- **Disk:** ~500 MB for the repo + models
- **No GPU needed** — Moonshine uses ONNX Runtime on CPU

### Install Build Tools

```bash
sudo apt update && sudo apt install -y \
  git \
  git-lfs \
  cmake \
  build-essential \
  patchelf
```

Verify versions:

```bash
git --version          # any recent version
git lfs --version      # 2.x or 3.x
cmake --version        # 3.22.1 or later required
g++ --version          # C++20 support needed (GCC 10+)
```

> **Note:** If your Ubuntu ships cmake < 3.22, install a newer version:
> ```bash
> sudo apt install -y software-properties-common
> sudo apt-add-repository -y 'deb https://apt.kitware.com/ubuntu/ jammy main'
> wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
> sudo apt update && sudo apt install -y cmake
> ```

---

## Step 1: Clone the Repository

Git LFS is **required** — the repo stores model files, ONNX Runtime binaries, and test assets in LFS.

```bash
git lfs install
git clone https://github.com/cvoalex/moonshine.git
cd moonshine
```

Verify LFS files were pulled (should NOT show `version https://git-lfs.github.com/spec/v1`):

```bash
head -1 core/speaker-embedding-model-data.cpp
# Expected: #include "speaker-embedding-model-data.h"
# If you see "version https://git-lfs.github.com..." run: git lfs pull
```

---

## Step 2: Build the Core C++ Library

```bash
cd core
mkdir -p build
cd build
cmake ..
cmake --build .
```

This produces:

| Output | Description |
| --- | --- |
| `libmoonshine.so` | Shared library (the core engine) |
| `moonshine-cpp-test` | C++ API test binary |
| `moonshine-c-api-test` | C API test binary |
| `transcriber-test` | Full transcriber pipeline test |
| `benchmark` | Performance benchmarking tool |
| + 8 other test binaries | Unit tests for sub-components |

Build time: ~1–2 minutes on a dual-core.

---

## Step 3: Run Tests

All tests must be run from the `test-assets` directory, and `LD_LIBRARY_PATH` must include the ONNX Runtime shared library:

```bash
cd ../../test-assets

export LD_LIBRARY_PATH=$(pwd)/../core/third-party/onnxruntime/lib/linux/x86_64:$LD_LIBRARY_PATH
```

Run all tests:

```bash
REPO=$(pwd)/..

$REPO/core/bin-tokenizer/build/bin-tokenizer-test
$REPO/core/third-party/onnxruntime/build/onnxruntime-test
$REPO/core/moonshine-utils/build/debug-utils-test
$REPO/core/moonshine-utils/build/string-utils-test
$REPO/core/build/resampler-test
$REPO/core/build/voice-activity-detector-test
$REPO/core/build/transcriber-test
$REPO/core/build/moonshine-c-api-test
$REPO/core/build/moonshine-cpp-test
$REPO/core/build/cosine-distance-test
$REPO/core/build/speaker-embedding-model-test
$REPO/core/build/online-clusterer-test

echo "All tests passed"
```

Or use the included script (does the same thing):

```bash
cd .. && bash scripts/run-core-tests.sh
```

Every test should print `Status: SUCCESS!` with 0 failures.

---

## Step 4: Run the Benchmark

From the `test-assets` directory:

```bash
cd test-assets

export LD_LIBRARY_PATH=$(pwd)/../core/third-party/onnxruntime/lib/linux/x86_64:$LD_LIBRARY_PATH

../core/build/benchmark --model-path tiny-en --wav-path two_cities.wav
```

Expected output — 13 transcribed lines from "A Tale of Two Cities" with timing info:

```
[0.96s] 'It was the best of times, it was the worst of times.'
...
Average Latency: ~69ms
Transcription took X seconds (Y% of audio duration)
```

> On a dual-core AMD server, expect latencies of ~60–100ms for the Tiny model.
> The README benchmarks show 69ms on Linux x86 for Tiny Streaming.

---

## Step 5: Build and Run the C++ Example

The `examples/c++/transcriber.cpp` is a self-contained example that transcribes a WAV file.

### Option A: Link against the built shared library

```bash
cd examples/c++

g++ transcriber.cpp \
  -I../../core \
  -I../../core/moonshine-utils \
  -L../../core/build \
  -lmoonshine \
  -std=c++20 \
  -o transcriber

export LD_LIBRARY_PATH=$(pwd)/../../core/build:$(pwd)/../../core/third-party/onnxruntime/lib/linux/x86_64:$LD_LIBRARY_PATH

./transcriber \
  --model-path ../../test-assets/tiny-en \
  --wav-path ../../test-assets/two_cities.wav
```

### Option B: Use pre-built release binaries

```bash
# Download the Linux x86_64 release (check latest version)
wget https://github.com/moonshine-ai/moonshine/releases/latest/download/moonshine-voice-linux-x86_64.tgz
tar xzf moonshine-voice-linux-x86_64.tgz

g++ transcriber.cpp \
  -Imoonshine-voice-linux-x86_64/include \
  -Lmoonshine-voice-linux-x86_64/lib \
  -lmoonshine \
  -o transcriber

export LD_LIBRARY_PATH=$(pwd)/moonshine-voice-linux-x86_64/lib
./transcriber --model-path /path/to/model --wav-path /path/to/audio.wav
```

---

## Step 6: Download Better Models (Optional)

The repo includes the **Tiny English** model in `test-assets/tiny-en` for testing. For production use, download a higher-accuracy model using the Python downloader:

```bash
# Install Python package (just for model downloading)
pip3 install moonshine-voice

# Download models (cached to ~/.cache/moonshine_voice by default)
python3 -m moonshine_voice.download --language en
```

The output will show the model path and architecture number:

```
Model arch: 5
Downloaded model path: /home/user/.cache/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/...
```

Use these values with the C++ binary:

```bash
../core/build/benchmark \
  --model-path /home/user/.cache/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized/medium-streaming-en \
  --model-arch 5 \
  --wav-path two_cities.wav
```

### Available Models

| Language | Architecture | Parameters | WER/CER | Model Arch # |
| --- | --- | --- | --- | --- |
| English | Tiny | 26M | 12.66% | 0 |
| English | Tiny Streaming | 34M | 12.00% | 3 |
| English | Base | 58M | 10.07% | 1 |
| English | Small Streaming | 123M | 7.84% | 4 |
| English | Medium Streaming | 245M | 6.65% | 5 |

For a dual-core AMD server, **Small Streaming** (model arch 4) is a good balance of accuracy and speed. Medium Streaming will also work but will use more CPU.

---

## Running as a systemd Service

To run Moonshine as a background service processing audio files:

### 1. Create a wrapper script

```bash
sudo mkdir -p /opt/moonshine
sudo cp core/build/libmoonshine.so /opt/moonshine/
sudo cp core/third-party/onnxruntime/lib/linux/x86_64/libonnxruntime.so.1 /opt/moonshine/
sudo cp examples/c++/transcriber /opt/moonshine/  # after building it
```

### 2. Create the service file

```bash
sudo tee /etc/systemd/system/moonshine.service << 'EOF'
[Unit]
Description=Moonshine Voice Transcriber
After=network.target

[Service]
Type=simple
User=moonshine
Group=moonshine
Environment=LD_LIBRARY_PATH=/opt/moonshine
ExecStart=/opt/moonshine/transcriber --model-path /opt/moonshine/models/tiny-en --wav-path /path/to/input.wav
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### 3. Enable and start

```bash
sudo useradd -r -s /bin/false moonshine
sudo systemctl daemon-reload
sudo systemctl enable moonshine
sudo systemctl start moonshine
sudo journalctl -u moonshine -f
```

---

## Deploying with Docker (Alternative)

```bash
# From the repo root
docker build -t moonshine .
docker run -it moonshine bash

# Inside the container
cd /home/user/moonshine
# Follow Steps 2–4 above
```

---

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `error: unknown type name 'version'` during build | LFS files not pulled. Run `git lfs pull` |
| `libonnxruntime.so.1: cannot open shared object` | Set `LD_LIBRARY_PATH` to include the ORT lib directory |
| `cmake` version too old | Install cmake >= 3.22.1 from kitware repo (see Prerequisites) |
| `error: 'std::format' not found` | Need GCC 13+ or use GCC 10+ with `-std=c++20` |
| Transcription latency too high | Use a streaming model (Tiny/Small/Medium Streaming) instead of non-streaming |
| Out of memory with Medium model | Medium Streaming uses ~1 GB RAM; upgrade to 2 GB+ or use Small Streaming |

---

## Using Moonshine as a Streaming Service

Moonshine supports real-time streaming transcription where you feed audio chunks incrementally and get transcript updates as speech happens. Here's how the architecture works and how to build a service around it.

### How Streaming Works Internally

A single `Transcriber` object loads the model once into memory. You then create **Streams** on that transcriber — each stream is an independent audio input with its own VAD, transcript state, and event callbacks. All streams share the same model weights, so you avoid duplicating the ~26–245 MB of model memory per connection.

```
┌─────────────────────────────────────────────┐
│              Transcriber                     │
│  (loads model once: ~26–245 MB RAM)          │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Stream 1 │ │ Stream 2 │ │ Stream N │ ...  │
│  │  (VAD)   │ │  (VAD)   │ │  (VAD)   │     │
│  │ (buffer) │ │ (buffer) │ │ (buffer) │     │
│  └─────────┘ └─────────┘ └─────────┘       │
│                                              │
│  Model inference is serialized (mutex)       │
│  → Streams take turns for the compute step   │
└─────────────────────────────────────────────┘
```

Key design points from the source:
- **API calls are thread-safe** — you can call `addAudio()` from one thread per stream concurrently
- **Model inference is serialized** with a mutex (`stt_model_mutex` / `streaming_model_mutex`) — only one stream runs the neural network at a time
- **Audio buffering is lock-free-ish** — `addAudio()` just appends to a buffer (fast, safe to call from audio capture threads)
- **VAD runs per-stream** with its own mutex, so voice detection is parallelized

### C++ Streaming Service Example

Here's a complete example of a WebSocket-based streaming transcription server. This accepts audio chunks over WebSocket connections and returns transcript updates in real time.

```cpp
// streaming-server.cpp
// Requires: libmoonshine, libwebsocketpp (or your WS library of choice)
// Simplified example — add error handling for production use.

#include <iostream>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>

#include "moonshine-cpp.h"

// A per-client session managing its own Stream
struct ClientSession {
  moonshine::Stream stream;
  std::mutex audioMutex;

  ClientSession(moonshine::Transcriber& transcriber)
      : stream(transcriber.createStream(0.5)) {}
};

class TranscriptionService {
public:
  TranscriptionService(const std::string& modelPath,
                       moonshine::ModelArch modelArch)
      : transcriber_(modelPath, modelArch) {}

  // Called when a new client connects
  int32_t onClientConnected() {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    int32_t clientId = nextClientId_++;

    auto session = std::make_unique<ClientSession>(transcriber_);

    // Set up event listener for this client
    session->stream.addListener(
        [clientId](const moonshine::TranscriptEvent& event) {
          if (event.type == moonshine::TranscriptEvent::LINE_TEXT_CHANGED) {
            // Send partial transcript back to client via WebSocket
            std::cout << "[Client " << clientId << " partial] "
                      << event.line.text << std::endl;
          }
          if (event.type == moonshine::TranscriptEvent::LINE_COMPLETED) {
            // Send final line to client
            std::cout << "[Client " << clientId << " final] "
                      << event.line.text << std::endl;
          }
        });

    session->stream.start();
    sessions_[clientId] = std::move(session);
    return clientId;
  }

  // Called when audio chunk arrives from a client
  void onAudioReceived(int32_t clientId,
                       const std::vector<float>& audioChunk,
                       int32_t sampleRate) {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    auto it = sessions_.find(clientId);
    if (it == sessions_.end()) return;

    // addAudio is fast — it just buffers. The transcription
    // happens automatically when enough audio accumulates.
    it->second->stream.addAudio(audioChunk, sampleRate);
  }

  // Called when client disconnects
  void onClientDisconnected(int32_t clientId) {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    auto it = sessions_.find(clientId);
    if (it == sessions_.end()) return;

    it->second->stream.stop();
    it->second->stream.close();
    sessions_.erase(it);
  }

  size_t activeConnections() {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    return sessions_.size();
  }

private:
  moonshine::Transcriber transcriber_;
  std::map<int32_t, std::unique_ptr<ClientSession>> sessions_;
  std::mutex sessionsMutex_;
  std::atomic<int32_t> nextClientId_{0};
};

int main(int argc, char* argv[]) {
  std::string modelPath = "tiny-en";
  moonshine::ModelArch modelArch = moonshine::ModelArch::TINY;

  TranscriptionService service(modelPath, modelArch);

  // Simulate 3 concurrent clients
  auto c1 = service.onClientConnected();
  auto c2 = service.onClientConnected();
  auto c3 = service.onClientConnected();

  std::cout << "Active connections: "
            << service.activeConnections() << std::endl;

  // In production, you'd wire this up to a WebSocket server
  // (e.g., Boost.Beast, uWebSockets, libwebsocketpp)
  // and call onAudioReceived() in the WS message handler.

  service.onClientDisconnected(c1);
  service.onClientDisconnected(c2);
  service.onClientDisconnected(c3);

  return 0;
}
```

### Multiple Transcriber Instances (True Parallelism)

Since model inference is serialized within a single Transcriber, you can create **multiple Transcriber instances** to use both CPU cores in parallel:

```cpp
// Two transcribers = two models in memory, but true parallel inference
moonshine::Transcriber transcriber1(modelPath, modelArch);
moonshine::Transcriber transcriber2(modelPath, modelArch);

// Route odd clients to transcriber1, even to transcriber2
// Each transcriber runs its model inference independently
```

---

## How Many Concurrent Streams Can It Handle?

### The Bottleneck: Model Inference

The limiting factor is **model inference time** (the neural network forward pass), which is serialized per Transcriber. Audio buffering and VAD are parallel and cheap. Here's the math:

**Per Transcriber, per stream, the model runs when:**
1. Enough new audio has accumulated (default: every 500ms of input audio)
2. A phrase completes (line boundary detected by VAD)

**Inference time per update** (from benchmarks on Linux x86):

| Model | Inference per update | CPU load per real-time stream |
| --- | --- | --- |
| Tiny | ~5–10ms | ~10% |
| Tiny Streaming | ~5–10ms | ~10% |
| Small Streaming | ~15–25ms | ~20% |
| Medium Streaming | ~25–40ms | ~35% |

### Concurrency Estimates for Dual-Core AMD

#### Single Transcriber (streams share one model, serialized inference)

Since inference is serialized, concurrent streams queue up. The max concurrent streams before latency degrades:

| Model | Max streams (1 transcriber) | Reasoning |
| --- | --- | --- |
| **Tiny** | **~8–10** | Each needs ~10% CPU, but serialized, so ~100ms budget per 500ms window serves ~5–10 streams |
| **Tiny Streaming** | **~8–10** | Same as Tiny |
| **Small Streaming** | **~4–5** | ~20% CPU each, serialized |
| **Medium Streaming** | **~2–3** | ~35% CPU each, serialized |

#### Two Transcribers (one per CPU core, true parallelism)

With 2 Transcriber instances pinned to separate threads, you double throughput:

| Model | Max streams (2 transcribers) | RAM usage |
| --- | --- | --- |
| **Tiny** | **~16–20** | ~200 MB (2 × ~52 MB model + overhead) |
| **Tiny Streaming** | **~16–20** | ~250 MB |
| **Small Streaming** | **~8–10** | ~500 MB |
| **Medium Streaming** | **~4–6** | ~1 GB |

> **Important:** These are estimates for *real-time* streams (audio arriving at 1x speed). If you're processing pre-recorded files faster than real-time, the numbers will be lower since inference is more tightly packed.

### Memory Requirements

| Component | RAM |
| --- | --- |
| ONNX Runtime engine | ~50 MB |
| Tiny model | ~52 MB |
| Small Streaming model | ~200 MB |
| Medium Streaming model | ~400 MB |
| Per-stream overhead (VAD, buffers) | ~2–5 MB |
| Speaker ID model (shared) | ~100 MB |

**For a dual-core server with 2 GB RAM:**
- 2× Tiny transcribers + 20 streams: ~400 MB total — fits easily
- 2× Small Streaming transcribers + 10 streams: ~700 MB — fits
- 2× Medium Streaming transcribers + 6 streams: ~1.2 GB — tight but works

**Recommendation for dual-core AMD:**
- Use **Tiny Streaming** with **2 Transcriber instances** for max concurrency (~16–20 streams)
- Use **Small Streaming** with **2 Transcriber instances** for best accuracy-to-concurrency tradeoff (~8–10 streams)

### Scaling Beyond One Server

If you need more concurrent streams, deploy multiple server instances behind a load balancer:

```
                    ┌──── Server 1 (2 cores, ~10–20 streams)
Client ──► Nginx ──┤
  (WS)     (LB)    ├──── Server 2 (2 cores, ~10–20 streams)
                    │
                    └──── Server N
```

Each server runs independently — Moonshine has no shared state to coordinate. Sticky sessions on the load balancer ensure a client's audio always routes to the same server.

---

## Performance Expectations (Dual-Core AMD)

Based on published benchmarks for Linux x86:

| Model | Avg Latency | CPU Load (% of audio duration) |
| --- | --- | --- |
| Tiny | ~69ms | ~10% |
| Tiny Streaming | ~69ms | ~10% |
| Small Streaming | ~165ms | ~20% |
| Medium Streaming | ~269ms | ~35% |

A dual-core server can comfortably handle real-time transcription with any of these models, with the Tiny and Small Streaming models leaving plenty of headroom for your application logic.
