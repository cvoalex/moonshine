/*
 * Moonshine Voice STT Server — AssemblyAI-compatible WebSocket API
 *
 * A standalone, zero-dependency C++ WebSocket server that speaks the
 * AssemblyAI Streaming v3 protocol using Moonshine Voice for local inference.
 *
 * Build:
 *   cmake --build core/build --target moonshine-server
 *
 * Run:
 *   ./core/build/moonshine-server --model-path ./test-assets/tiny-en \
 *       --model-arch 0 --port 8765
 *
 * Protocol (AssemblyAI v3 compatible):
 *   Client → Server:
 *     • Raw PCM16-LE audio bytes (binary WebSocket frames)
 *     • {"type":"ForceEndpoint"}   — force end-of-turn
 *     • {"type":"Terminate"}       — end session
 *
 *   Server → Client:
 *     • {"type":"Begin","id":"...","expires_at":...}
 *     • {"type":"Turn","turn_order":N,"transcript":"...","end_of_turn":...}
 *     • {"type":"Termination","audio_duration_seconds":...,"session_duration_seconds":...}
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "moonshine-cpp.h"

// ─── Minimal SHA-1 (for WebSocket handshake) ───────────────────────────

namespace {

struct SHA1 {
  uint32_t h[5];
  uint64_t totalBits;
  uint8_t buf[64];
  size_t bufLen;

  SHA1() { reset(); }

  void reset() {
    h[0] = 0x67452301;
    h[1] = 0xEFCDAB89;
    h[2] = 0x98BADCFE;
    h[3] = 0x10325476;
    h[4] = 0xC3D2E1F0;
    totalBits = 0;
    bufLen = 0;
  }

  static uint32_t rol(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
  }

  void processBlock(const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
      w[i] = (uint32_t(block[i * 4]) << 24) |
             (uint32_t(block[i * 4 + 1]) << 16) |
             (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
    }
    for (int i = 16; i < 80; i++) {
      w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
      uint32_t f, k;
      if (i < 20) {
        f = (b & c) | ((~b) & d);
        k = 0x5A827999;
      } else if (i < 40) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1;
      } else if (i < 60) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDC;
      } else {
        f = b ^ c ^ d;
        k = 0xCA62C1D6;
      }
      uint32_t temp = rol(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rol(b, 30);
      b = a;
      a = temp;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
  }

  void update(const void *data, size_t len) {
    auto *p = static_cast<const uint8_t *>(data);
    totalBits += len * 8;
    while (len > 0) {
      size_t toCopy = std::min(len, size_t(64) - bufLen);
      std::memcpy(buf + bufLen, p, toCopy);
      bufLen += toCopy;
      p += toCopy;
      len -= toCopy;
      if (bufLen == 64) {
        processBlock(buf);
        bufLen = 0;
      }
    }
  }

  void digest(uint8_t out[20]) {
    buf[bufLen++] = 0x80;
    if (bufLen > 56) {
      while (bufLen < 64) buf[bufLen++] = 0;
      processBlock(buf);
      bufLen = 0;
    }
    while (bufLen < 56) buf[bufLen++] = 0;
    for (int i = 7; i >= 0; i--) {
      buf[56 + (7 - i)] = uint8_t(totalBits >> (i * 8));
    }
    processBlock(buf);
    for (int i = 0; i < 5; i++) {
      out[i * 4 + 0] = uint8_t(h[i] >> 24);
      out[i * 4 + 1] = uint8_t(h[i] >> 16);
      out[i * 4 + 2] = uint8_t(h[i] >> 8);
      out[i * 4 + 3] = uint8_t(h[i]);
    }
  }
};

// ─── Base64 encode ────────────────────────────────────────────────────

std::string base64Encode(const uint8_t *data, size_t len) {
  static const char table[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve(((len + 2) / 3) * 4);
  for (size_t i = 0; i < len; i += 3) {
    uint32_t n = uint32_t(data[i]) << 16;
    if (i + 1 < len) n |= uint32_t(data[i + 1]) << 8;
    if (i + 2 < len) n |= uint32_t(data[i + 2]);
    out += table[(n >> 18) & 0x3F];
    out += table[(n >> 12) & 0x3F];
    out += (i + 1 < len) ? table[(n >> 6) & 0x3F] : '=';
    out += (i + 2 < len) ? table[n & 0x3F] : '=';
  }
  return out;
}

// ─── UUID v4 ─────────────────────────────────────────────────────────

std::string generateUUID() {
  static std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<uint32_t> dist(0, 15);
  const char hex[] = "0123456789abcdef";
  std::string uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";
  for (auto &c : uuid) {
    if (c == 'x') {
      c = hex[dist(gen)];
    } else if (c == 'y') {
      c = hex[(dist(gen) & 0x3) | 0x8];
    }
  }
  return uuid;
}

// ─── WebSocket framing ───────────────────────────────────────────────

// Read exactly n bytes from fd. Returns false on error/close.
bool readExact(int fd, void *buf, size_t n) {
  auto *p = static_cast<uint8_t *>(buf);
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r <= 0) return false;
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

// Send all bytes. Returns false on error.
bool sendAll(int fd, const void *buf, size_t n) {
  auto *p = static_cast<const uint8_t *>(buf);
  while (n > 0) {
    ssize_t w = ::write(fd, p, n);
    if (w <= 0) return false;
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

// Read one WebSocket frame. opcode: 1=text, 2=binary, 8=close, 9=ping, 10=pong
struct WSFrame {
  uint8_t opcode;
  bool fin;
  std::vector<uint8_t> payload;
};

bool wsReadFrame(int fd, WSFrame &frame) {
  uint8_t hdr[2];
  if (!readExact(fd, hdr, 2)) return false;
  frame.fin = (hdr[0] & 0x80) != 0;
  frame.opcode = hdr[0] & 0x0F;
  bool masked = (hdr[1] & 0x80) != 0;
  uint64_t payloadLen = hdr[1] & 0x7F;
  if (payloadLen == 126) {
    uint8_t ext[2];
    if (!readExact(fd, ext, 2)) return false;
    payloadLen = (uint64_t(ext[0]) << 8) | ext[1];
  } else if (payloadLen == 127) {
    uint8_t ext[8];
    if (!readExact(fd, ext, 8)) return false;
    payloadLen = 0;
    for (int i = 0; i < 8; i++) payloadLen = (payloadLen << 8) | ext[i];
  }
  uint8_t mask[4] = {};
  if (masked) {
    if (!readExact(fd, mask, 4)) return false;
  }
  frame.payload.resize(static_cast<size_t>(payloadLen));
  if (payloadLen > 0) {
    if (!readExact(fd, frame.payload.data(), frame.payload.size()))
      return false;
    if (masked) {
      for (size_t i = 0; i < frame.payload.size(); i++) {
        frame.payload[i] ^= mask[i % 4];
      }
    }
  }
  return true;
}

bool wsSendFrame(int fd, uint8_t opcode, const void *data, size_t len) {
  uint8_t hdr[10];
  size_t hdrLen = 0;
  hdr[0] = 0x80 | opcode;  // FIN + opcode
  if (len < 126) {
    hdr[1] = uint8_t(len);
    hdrLen = 2;
  } else if (len < 65536) {
    hdr[1] = 126;
    hdr[2] = uint8_t(len >> 8);
    hdr[3] = uint8_t(len);
    hdrLen = 4;
  } else {
    hdr[1] = 127;
    for (int i = 0; i < 8; i++) {
      hdr[2 + i] = uint8_t(len >> ((7 - i) * 8));
    }
    hdrLen = 10;
  }
  if (!sendAll(fd, hdr, hdrLen)) return false;
  if (len > 0 && !sendAll(fd, data, len)) return false;
  return true;
}

bool wsSendText(int fd, const std::string &text) {
  return wsSendFrame(fd, 1, text.data(), text.size());
}

bool wsSendClose(int fd) { return wsSendFrame(fd, 8, nullptr, 0); }

// ─── WebSocket HTTP upgrade handshake ─────────────────────────────────

bool wsHandshake(int fd) {
  // Read HTTP request (up to 4KB)
  char reqBuf[4096];
  size_t reqLen = 0;
  while (reqLen < sizeof(reqBuf) - 1) {
    ssize_t r = ::read(fd, reqBuf + reqLen, sizeof(reqBuf) - 1 - reqLen);
    if (r <= 0) return false;
    reqLen += static_cast<size_t>(r);
    reqBuf[reqLen] = '\0';
    if (std::strstr(reqBuf, "\r\n\r\n")) break;
  }

  // Find Sec-WebSocket-Key
  const char *keyHeader = "Sec-WebSocket-Key:";
  const char *keyStart = strcasestr(reqBuf, keyHeader);
  if (!keyStart) return false;
  keyStart += std::strlen(keyHeader);
  while (*keyStart == ' ') keyStart++;
  const char *keyEnd = std::strstr(keyStart, "\r\n");
  if (!keyEnd) return false;
  std::string clientKey(keyStart, keyEnd);

  // Compute accept key: SHA1(clientKey + magic) → base64
  std::string concat = clientKey + "258EAFA5-E914-47DA-95CA-5AB5DC175B07";
  SHA1 sha;
  sha.update(concat.data(), concat.size());
  uint8_t digest[20];
  sha.digest(digest);
  std::string acceptKey = base64Encode(digest, 20);

  // Send HTTP 101
  std::string response = "HTTP/1.1 101 Switching Protocols\r\n"
                         "Upgrade: websocket\r\n"
                         "Connection: Upgrade\r\n"
                         "Sec-WebSocket-Accept: " +
                         acceptKey + "\r\n\r\n";
  return sendAll(fd, response.data(), response.size());
}

// ─── JSON helpers (minimal, no external library) ──────────────────────

std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

// Simple JSON string value extractor: returns value for "key":"value"
std::string jsonGetString(const std::string &json, const std::string &key) {
  std::string search = "\"" + key + "\"";
  auto pos = json.find(search);
  if (pos == std::string::npos) return "";
  pos = json.find(':', pos + search.size());
  if (pos == std::string::npos) return "";
  pos = json.find('"', pos + 1);
  if (pos == std::string::npos) return "";
  auto endPos = json.find('"', pos + 1);
  if (endPos == std::string::npos) return "";
  return json.substr(pos + 1, endPos - pos - 1);
}

// ─── PCM16 → float conversion ───────────────────────────────────────

std::vector<float> pcm16ToFloat(const uint8_t *data, size_t bytes) {
  size_t nSamples = bytes / 2;
  std::vector<float> out(nSamples);
  for (size_t i = 0; i < nSamples; i++) {
    int16_t sample =
        static_cast<int16_t>(data[i * 2] | (data[i * 2 + 1] << 8));
    out[i] = static_cast<float>(sample) / 32768.0f;
  }
  return out;
}

}  // namespace

// ─── Per-client session ─────────────────────────────────────────────

static void handleClient(int clientFd, moonshine::Transcriber &transcriber,
                          int32_t sampleRate, double updateInterval) {
  // Enable TCP_NODELAY for low-latency responses
  int flag = 1;
  setsockopt(clientFd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

  // WebSocket handshake
  if (!wsHandshake(clientFd)) {
    std::cerr << "WebSocket handshake failed" << std::endl;
    close(clientFd);
    return;
  }

  std::string sessionId = generateUUID();
  auto sessionStart = std::chrono::steady_clock::now();
  uint64_t totalSamples = 0;
  std::atomic<int> turnOrder{0};

  // Mutex to protect writing to the socket from multiple threads
  std::mutex writeMutex;

  std::cout << "[" << sessionId.substr(0, 8) << "] Client connected"
            << std::endl;

  // Send Begin message
  {
    auto now = std::chrono::system_clock::now();
    auto expiresAt =
        std::chrono::duration_cast<std::chrono::seconds>(
            (now + std::chrono::hours(24)).time_since_epoch())
            .count();
    std::string beginMsg =
        "{\"type\":\"Begin\",\"id\":\"" + sessionId +
        "\",\"expires_at\":" + std::to_string(expiresAt) + "}";
    wsSendText(clientFd, beginMsg);
  }

  // Create a stream for this session
  moonshine::Stream stream = transcriber.createStream(updateInterval);

  // Listener that sends Turn messages to the client
  class SessionListener : public moonshine::TranscriptEventListener {
   public:
    int clientFd;
    std::atomic<int> *turnOrder;
    std::mutex *writeMutex;

    void onLineTextChanged(const moonshine::LineTextChanged &event) override {
      int order = ++(*turnOrder);
      std::string msg =
          "{\"type\":\"Turn\""
          ",\"turn_order\":" +
          std::to_string(order) +
          ",\"turn_is_formatted\":false"
          ",\"end_of_turn\":false"
          ",\"end_of_turn_confidence\":0.0"
          ",\"transcript\":\"" +
          jsonEscape(event.line.text) +
          "\""
          ",\"words\":[]}";
      std::lock_guard<std::mutex> lock(*writeMutex);
      wsSendText(clientFd, msg);
    }

    void onLineCompleted(const moonshine::LineCompleted &event) override {
      int order = ++(*turnOrder);
      std::string msg =
          "{\"type\":\"Turn\""
          ",\"turn_order\":" +
          std::to_string(order) +
          ",\"turn_is_formatted\":true"
          ",\"end_of_turn\":true"
          ",\"end_of_turn_confidence\":1.0"
          ",\"transcript\":\"" +
          jsonEscape(event.line.text) +
          "\""
          ",\"words\":[]}";
      std::lock_guard<std::mutex> lock(*writeMutex);
      wsSendText(clientFd, msg);
    }
  };

  SessionListener listener;
  listener.clientFd = clientFd;
  listener.turnOrder = &turnOrder;
  listener.writeMutex = &writeMutex;
  stream.addListener(&listener);
  stream.start();

  // Read frames
  WSFrame frame;
  bool running = true;
  while (running && wsReadFrame(clientFd, frame)) {
    switch (frame.opcode) {
      case 2: {  // Binary — PCM16 audio
        if (frame.payload.empty()) break;
        auto floats = pcm16ToFloat(frame.payload.data(), frame.payload.size());
        totalSamples += floats.size();
        stream.addAudio(floats, sampleRate);
        break;
      }
      case 1: {  // Text — JSON control message
        std::string text(frame.payload.begin(), frame.payload.end());
        std::string msgType = jsonGetString(text, "type");
        if (msgType == "ForceEndpoint") {
          stream.updateTranscription(moonshine::Stream::FLAG_FORCE_UPDATE);
        } else if (msgType == "Terminate") {
          running = false;
        }
        break;
      }
      case 8: {  // Close
        running = false;
        break;
      }
      case 9: {  // Ping → Pong
        std::lock_guard<std::mutex> lock(writeMutex);
        wsSendFrame(clientFd, 10, frame.payload.data(), frame.payload.size());
        break;
      }
      default:
        break;
    }
  }

  // Stop stream and flush final events
  try {
    stream.stop();
  } catch (const std::exception &e) {
    std::cerr << "[" << sessionId.substr(0, 8)
              << "] Stream stop error: " << e.what() << std::endl;
  }

  // Send Termination
  auto sessionEnd = std::chrono::steady_clock::now();
  double audioDuration =
      static_cast<double>(totalSamples) / static_cast<double>(sampleRate);
  double sessionDuration =
      std::chrono::duration<double>(sessionEnd - sessionStart).count();

  {
    std::ostringstream oss;
    oss << "{\"type\":\"Termination\""
        << ",\"audio_duration_seconds\":" << audioDuration
        << ",\"session_duration_seconds\":" << sessionDuration << "}";
    std::lock_guard<std::mutex> lock(writeMutex);
    wsSendText(clientFd, oss.str());
    wsSendClose(clientFd);
  }

  stream.close();
  close(clientFd);

  std::cout << "[" << sessionId.substr(0, 8) << "] Disconnected — audio="
            << audioDuration << "s, session=" << sessionDuration << "s"
            << std::endl;
}

// ─── Main ──────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  std::string modelPath;
  int modelArchInt = -1;
  int port = 8765;
  int32_t sampleRate = 16000;
  double updateInterval = 0.5;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if ((arg == "--model-path" || arg == "-m") && i + 1 < argc) {
      modelPath = argv[++i];
    } else if ((arg == "--model-arch" || arg == "-a") && i + 1 < argc) {
      modelArchInt = std::stoi(argv[++i]);
    } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
      port = std::stoi(argv[++i]);
    } else if (arg == "--sample-rate" && i + 1 < argc) {
      sampleRate = std::stoi(argv[++i]);
    } else if (arg == "--update-interval" && i + 1 < argc) {
      updateInterval = std::stod(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Moonshine STT Server — AssemblyAI-compatible WebSocket API\n\n"
          << "Usage:\n"
          << "  moonshine-server --model-path <path> --model-arch <int> "
             "[options]\n\n"
          << "Options:\n"
          << "  -m, --model-path <path>    Path to model directory (required)\n"
          << "  -a, --model-arch <int>     Model arch: 0=tiny, 1=base, "
             "2=tiny-streaming,\n"
          << "                             3=base-streaming, 4=small-streaming, "
             "5=medium-streaming\n"
          << "  -p, --port <int>           Port (default: 8765)\n"
          << "  --sample-rate <int>        Expected sample rate (default: "
             "16000)\n"
          << "  --update-interval <float>  Transcription update interval in "
             "seconds (default: 0.5)\n"
          << "  -h, --help                 Show this help\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (modelPath.empty() || modelArchInt < 0) {
    std::cerr
        << "Error: --model-path and --model-arch are required.\n"
        << "Run with --help for usage.\n";
    return 1;
  }

  auto modelArch = static_cast<moonshine::ModelArch>(modelArchInt);

  std::cout << "Loading model: path=" << modelPath << ", arch=" << modelArchInt
            << std::endl;
  moonshine::Transcriber transcriber(modelPath, modelArch, updateInterval);
  std::cout << "Model loaded successfully" << std::endl;

  // Create listening socket
  int serverFd = socket(AF_INET, SOCK_STREAM, 0);
  if (serverFd < 0) {
    std::perror("socket");
    return 1;
  }

  int opt = 1;
  setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port));

  if (bind(serverFd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) <
      0) {
    std::perror("bind");
    close(serverFd);
    return 1;
  }

  if (listen(serverFd, 8) < 0) {
    std::perror("listen");
    close(serverFd);
    return 1;
  }

  std::cout << "Moonshine STT server listening on ws://0.0.0.0:" << port
            << std::endl;
  std::cout << "  sample_rate=" << sampleRate
            << ", update_interval=" << updateInterval << "s" << std::endl;
  std::cout << "  Protocol: AssemblyAI Streaming v3 compatible" << std::endl;
  std::cout << std::endl;

  // Accept loop — one thread per client
  while (true) {
    struct sockaddr_in clientAddr {};
    socklen_t clientLen = sizeof(clientAddr);
    int clientFd = accept(serverFd,
                          reinterpret_cast<struct sockaddr *>(&clientAddr),
                          &clientLen);
    if (clientFd < 0) {
      std::perror("accept");
      continue;
    }
    // Detach a thread for each client
    std::thread(handleClient, clientFd, std::ref(transcriber), sampleRate,
                updateInterval)
        .detach();
  }

  close(serverFd);
  return 0;
}
