#include "driver/gpio.h"
#include "esp_camera.h"
#include "esp_timer.h"
#include "fb_gfx.h"
#include "img_converters.h"
#include "soc/rtc_cntl_reg.h" //disable brownout problems
#include "soc/soc.h"          //disable brownout problems
#include <ArduinoWebsockets.h>
#include <WiFi.h>


// AI-Thinker ESP32-CAM pin definitions
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27

#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22
#define WIFI_UNAME "SWARNO"
#define WIFI_PASS "snag@1733"
// ============== OPTIMIZED SETTINGS ==============
#define STREAM_FRAMESIZE FRAMESIZE_QVGA // 320x240
#define STREAM_QUALITY                                                         \
  30 // Lower quality (higher number) = smaller packet size, preventing "Corrupt
     // JPEG"
#define CAPTURE_FRAMESIZE FRAMESIZE_VGA // 640x480
#define CAPTURE_QUALITY 8             // Good balance for recognition
#define RESUME_TIMEOUT_MS 15000         // Auto-resume after 15 seconds
// ================================================

// ============== STATE MACHINE ==============
enum CameraState {
  STATE_STREAMING, // Normal QVGA streaming
  STATE_PAUSED,    // Streaming paused, switching to VGA
  STATE_CAPTURING, // Capturing VGA frame
  STATE_WAITING    // Waiting for RESUME_STREAM command
};

CameraState currentState = STATE_STREAMING;
unsigned long pauseStartTime = 0;
// ===========================================

char *url = "ws://192.168.137.1:3000/ws";

using namespace websockets;
WebsocketsClient client;

///////////////////////////////////CALLBACK
///FUNCTIONS///////////////////////////////////
void onMessageCallback(WebsocketsMessage message) {
  String cmd = message.data();
  Serial.print("Got Message: ");
  Serial.println(cmd);

  if (cmd == "CAPTURE_HQ") {
    if (currentState == STATE_STREAMING) {
      currentState = STATE_PAUSED;
      pauseStartTime = millis();
      Serial.println(">>> Stream PAUSED - Switching to VGA capture");
    }
  } else if (cmd == "RESUME_STREAM") {
    if (currentState == STATE_WAITING) {
      currentState = STATE_STREAMING;
      Serial.println(">>> Stream RESUMED by server");
    }
  }
}

///////////////////////////////////INITIALIZE
///FUNCTIONS///////////////////////////////////
esp_err_t init_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 8000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = STREAM_QUALITY;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    Serial.println("PSRAM found - dual buffer mode");
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = STREAM_QUALITY;
    config.fb_count = 1;
    Serial.println("No PSRAM - single buffer mode");
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return err;
  }

  // sensor_t *s = esp_camera_sensor_get();
  // s->set_framesize(s, STREAM_FRAMESIZE);
  // s->set_quality(s, STREAM_QUALITY);

  Serial.println("Camera initialized in STREAMING mode (QVGA 320x240)");
  return ESP_OK;
};

esp_err_t init_wifi() {
  WiFi.begin(WIFI_UNAME, WIFI_PASS);
  Serial.println("Starting Wifi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  Serial.println("Connecting to websocket");
  client.onMessage(onMessageCallback);
  bool connected = client.connect(url);
  if (!connected) {
    Serial.println("Cannot connect to websocket server!");
    return ESP_FAIL;
  }

  Serial.println("Websocket Connected!");
  client.send("deviceId");
  return ESP_OK;
};

///////////////////////////////////SETUP///////////////////////////////////
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n=== ESP32-CAM State Machine Streaming ===");

  init_camera();
  init_wifi();
}

///////////////////////////////////MAIN LOOP///////////////////////////////////
void loop() {
  if (!client.available())
    return;

  client.poll(); // Always check for messages

  switch (currentState) {

  case STATE_STREAMING: {
    // Normal QVGA streaming
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      ESP.restart();
    }
    client.sendBinary((const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    break;
  }

  case STATE_PAUSED: {
    // Switch camera to VGA mode
    Serial.println("Switching to HQ mode...");
    sensor_t *s = esp_camera_sensor_get();
    // s->set_framesize(s, CAPTURE_FRAMESIZE);
    s->set_quality(s, CAPTURE_QUALITY);
    delay(150); // Allow sensor to stabilize
    currentState = STATE_CAPTURING;
    break;
  }

  case STATE_CAPTURING: {
    // Capture and send VGA frame
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) {
      // Send marker before HQ frame so server knows it's HQ
      client.send("HQ_FRAME_START");
      client.sendBinary((const char *)fb->buf, fb->len);
      Serial.printf("HQ frame sent: %d bytes \n", fb->len);
      esp_camera_fb_return(fb);
    }

    // Switch back to QVGA but don't stream yet
    sensor_t *s = esp_camera_sensor_get();
    // s->set_framesize(s, STREAM_FRAMESIZE);
    s->set_quality(s, STREAM_QUALITY);

    currentState = STATE_WAITING;
    Serial.println("Waiting for RESUME_STREAM...");
    break;
  }

  case STATE_WAITING: {
    // Check for timeout - auto-resume after 15 seconds
    if (millis() - pauseStartTime > RESUME_TIMEOUT_MS) {
      Serial.println(">>> TIMEOUT - Auto-resuming stream");
      currentState = STATE_STREAMING;
    }
    // Just poll for messages, don't stream
    delay(50);
    break;
  }
  }
}