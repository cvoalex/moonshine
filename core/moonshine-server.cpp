/*
 * Moonshine Voice STT Server — AssemblyAI-compatible WebSocket API
 *
 * Uses IXWebSocket library for proper, standards-compliant WebSocket handling.
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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>

#include "moonshine-cpp.h"

namespace {

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

// ─── Per-client session ─────────────────────────────────────────────

struct Session {
  std::string id;
  std::chrono::steady_clock::time_point startTime;
  std::unique_ptr<moonshine::Stream> stream;
  std::atomic<int> turnOrder{0};
  uint64_t totalSamples{0};
  int32_t sampleRate{16000};

  // Listener for stream transcript events → sends Turn messages via WebSocket
  struct Listener : public moonshine::TranscriptEventListener {
    ix::WebSocket *ws{nullptr};
    std::atomic<int> *turnOrderPtr{nullptr};

    void onLineTextChanged(const moonshine::LineTextChanged &event) override {
      if (!ws) return;
      int order = ++(*turnOrderPtr);
      std::cout << "  [EVENT] LineTextChanged: \"" << event.line.text << "\""
                << std::endl;
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
      std::cout << "  [SEND] " << msg << std::endl;
      ws->send(msg);
    }

    void onLineCompleted(const moonshine::LineCompleted &event) override {
      if (!ws) return;
      int order = ++(*turnOrderPtr);
      std::cout << "  [EVENT] LineCompleted: \"" << event.line.text << "\""
                << std::endl;
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
      std::cout << "  [SEND] " << msg << std::endl;
      ws->send(msg);
    }
  };

  Listener listener;
};

}  // namespace

// ─── Main ──────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  std::setbuf(stdout, nullptr);
  std::setbuf(stderr, nullptr);

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

  // Initialize IXWebSocket networking (no-op on Unix, needed on Windows)
  ix::initNetSystem();

  // Session registry — keyed by IXWebSocket connection ID
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions;
  std::mutex sessionsMutex;

  ix::WebSocketServer server(port, "0.0.0.0");
  server.disablePerMessageDeflate();

  server.setOnClientMessageCallback(
      [&transcriber, &sessions, &sessionsMutex, sampleRate, updateInterval](
          std::shared_ptr<ix::ConnectionState> connectionState,
          ix::WebSocket &ws, const ix::WebSocketMessagePtr &msg) {
        std::string connId = connectionState->getId();

        switch (msg->type) {
          case ix::WebSocketMessageType::Open: {
            auto session = std::make_shared<Session>();
            session->id = generateUUID();
            session->startTime = std::chrono::steady_clock::now();
            session->sampleRate = sampleRate;
            session->stream = std::make_unique<moonshine::Stream>(
                transcriber.createStream(updateInterval));

            // Wire up the listener
            session->listener.ws = &ws;
            session->listener.turnOrderPtr = &session->turnOrder;
            session->stream->addListener(&session->listener);
            session->stream->start();

            {
              std::lock_guard<std::mutex> lock(sessionsMutex);
              sessions[connId] = session;
            }

            // Send Begin message
            auto now = std::chrono::system_clock::now();
            auto expiresAt =
                std::chrono::duration_cast<std::chrono::seconds>(
                    (now + std::chrono::hours(24)).time_since_epoch())
                    .count();
            std::string beginMsg =
                "{\"type\":\"Begin\",\"id\":\"" + session->id +
                "\",\"expires_at\":" + std::to_string(expiresAt) + "}";

            std::cout << "[" << session->id.substr(0, 8) << "] Client connected"
                      << std::endl;
            std::cout << "  [SEND] " << beginMsg << std::endl;
            ws.send(beginMsg);
            break;
          }

          case ix::WebSocketMessageType::Message: {
            std::shared_ptr<Session> session;
            {
              std::lock_guard<std::mutex> lock(sessionsMutex);
              auto it = sessions.find(connId);
              if (it != sessions.end()) session = it->second;
            }
            if (!session) break;

            if (msg->binary) {
              // Binary — PCM16 audio
              const auto *data =
                  reinterpret_cast<const uint8_t *>(msg->str.data());
              size_t bytes = msg->str.size();
              if (bytes == 0) break;
              auto floats = pcm16ToFloat(data, bytes);
              session->totalSamples += floats.size();
              std::cout << "  [AUDIO] " << floats.size() << " samples ("
                        << (double(floats.size()) / sampleRate * 1000.0)
                        << "ms), total="
                        << (double(session->totalSamples) / sampleRate) << "s"
                        << std::endl;
              session->stream->addAudio(floats, sampleRate);
            } else {
              // Text — JSON control message
              std::string msgType = jsonGetString(msg->str, "type");
              std::cout << "  [CTRL] " << msg->str << std::endl;

              if (msgType == "ForceEndpoint") {
                std::cout << "  [CTRL] Forcing transcription update"
                          << std::endl;
                session->stream->updateTranscription(
                    moonshine::Stream::FLAG_FORCE_UPDATE);
              } else if (msgType == "Terminate") {
                std::cout << "  [CTRL] Terminate requested" << std::endl;

                // Stop stream and flush final events
                try {
                  session->stream->stop();
                } catch (const std::exception &e) {
                  std::cout << "Stream stop error: " << e.what() << std::endl;
                }

                // Send Termination
                auto sessionEnd = std::chrono::steady_clock::now();
                double audioDur =
                    double(session->totalSamples) / double(sampleRate);
                double sessDur =
                    std::chrono::duration<double>(sessionEnd -
                                                  session->startTime)
                        .count();
                std::ostringstream oss;
                oss << "{\"type\":\"Termination\""
                    << ",\"audio_duration_seconds\":" << audioDur
                    << ",\"session_duration_seconds\":" << sessDur << "}";
                std::cout << "  [SEND] " << oss.str() << std::endl;
                ws.send(oss.str());

                session->stream->close();
                session->listener.ws = nullptr;
                {
                  std::lock_guard<std::mutex> lock(sessionsMutex);
                  sessions.erase(connId);
                }
                ws.close();
              }
            }
            break;
          }

          case ix::WebSocketMessageType::Close: {
            std::shared_ptr<Session> session;
            {
              std::lock_guard<std::mutex> lock(sessionsMutex);
              auto it = sessions.find(connId);
              if (it != sessions.end()) {
                session = it->second;
                sessions.erase(it);
              }
            }
            if (!session) break;

            session->listener.ws = nullptr;
            try {
              session->stream->stop();
            } catch (const std::exception &e) {
              std::cout << "Stream stop error: " << e.what() << std::endl;
            }

            auto sessionEnd = std::chrono::steady_clock::now();
            double audioDur =
                double(session->totalSamples) / double(sampleRate);
            double sessDur =
                std::chrono::duration<double>(sessionEnd - session->startTime)
                    .count();
            session->stream->close();

            std::cout << "[" << session->id.substr(0, 8)
                      << "] Disconnected — audio=" << audioDur
                      << "s, session=" << sessDur << "s" << std::endl;
            break;
          }

          case ix::WebSocketMessageType::Error: {
            std::cout << "[" << connId.substr(0, 8)
                      << "] WebSocket error: " << msg->errorInfo.reason
                      << std::endl;
            break;
          }

          default:
            break;
        }
      });

  auto res = server.listen();
  if (!res.first) {
    std::cerr << "Failed to listen on port " << port << ": " << res.second
              << std::endl;
    ix::uninitNetSystem();
    return 1;
  }

  server.start();

  std::cout << "Moonshine STT server listening on ws://0.0.0.0:" << port
            << std::endl;
  std::cout << "  sample_rate=" << sampleRate
            << ", update_interval=" << updateInterval << "s" << std::endl;
  std::cout << "  Protocol: AssemblyAI Streaming v3 compatible" << std::endl;
  std::cout << std::endl;

  // Block until interrupted
  server.wait();

  ix::uninitNetSystem();
  return 0;
}
