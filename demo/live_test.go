package main

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func testLiveApp(t *testing.T) *app {
	t.Helper()
	dir := t.TempDir()
	pipeline := filepath.Join(dir, "worker")
	// Exercise HTTP/worker lifecycle without GPU models; real rolling inference
	// is checked separately against the pinned video reference.
	script := `#!/bin/sh
printf '%s' "${12}" > "${11}/detect-interval"
printf 'READY\n'
n=0
while read command; do
 printf '%s\n' "$command" > "${11}/last-command"
 n=$((n+1))
 if test "$n" = 1; then printf 'WARMUP 1\n'; else
  printf GEMPOSE2 > "${11}/pose.gpose"
  printf 'POSE 1\n'
 fi
done
`
	if err := os.WriteFile(pipeline, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	a := &app{cfg: config{data: dir, pipeline: pipeline, threads: 8}}
	t.Cleanup(func() { a.stopLive() })
	return a
}
func TestLiveDetectionIntervalValidation(t *testing.T) {
	a := testLiveApp(t)
	for _, value := range []string{"0", "31", "1.5", "nope", "-1"} {
		if w := liveRequest(a, "POST", "/api/live?detect_interval="+value, nil); w.Code != 400 {
			t.Fatalf("interval %s: status %d", value, w.Code)
		}
	}
	if w := liveRequest(a, "POST", "/api/live?detect_interval=7", nil); w.Code != 201 {
		t.Fatalf("valid interval: status %d: %s", w.Code, w.Body.String())
	}
	data, err := os.ReadFile(filepath.Join(a.live.dir, "detect-interval"))
	if err != nil || string(data) != "7" {
		t.Fatalf("worker interval: %q, %v", data, err)
	}
}
func liveRequest(a *app, method, path string, data []byte) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	a.route(w, httptest.NewRequest(method, path, bytes.NewReader(data)))
	return w
}
func startTestLive(t *testing.T, a *app) string {
	t.Helper()
	w := liveRequest(a, "POST", "/api/live", nil)
	if w.Code != 201 {
		t.Fatalf("start: %d %s", w.Code, w.Body)
	}
	var session struct{ ID string }
	if err := json.Unmarshal(w.Body.Bytes(), &session); err != nil {
		t.Fatal(err)
	}
	return "/api/live/" + session.ID
}
func livePNG(t *testing.T, width int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewRGBA(image.Rect(0, 0, width, 16))); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}
func TestLiveLifecycle(t *testing.T) {
	a := testLiveApp(t)
	path := startTestLive(t, a)
	if w := liveRequest(a, "POST", "/api/live", nil); w.Code != 409 {
		t.Fatalf("second session: %d", w.Code)
	}
	if w := liveRequest(a, "PUT", path+"/frame", []byte("bad")); w.Code != 400 {
		t.Fatalf("bad image: %d", w.Code)
	}
	frame := livePNG(t, 16)
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 204 || w.Header().Get("X-GEMX-Sequence") != "1" {
		t.Fatalf("warmup: %d %s", w.Code, w.Body)
	}
	if w := liveRequest(a, "PUT", path+"/frame", livePNG(t, 24)); w.Code != 409 {
		t.Fatalf("dimension change: %d", w.Code)
	}
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 200 || w.Body.String() != "GEMPOSE2" || w.Header().Get("X-GEMX-Sequence") != "2" {
		t.Fatalf("pose: %d %s", w.Code, w.Body)
	}
	a.liveMu.Lock()
	s := a.live
	a.liveMu.Unlock()
	s.mu.Lock()
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 409 {
		t.Fatalf("overlap: %d", w.Code)
	}
	s.mu.Unlock()
	if w := liveRequest(a, "DELETE", path, nil); w.Code != 204 {
		t.Fatalf("stop: %d", w.Code)
	}
	if _, err := os.Stat(s.dir); !os.IsNotExist(err) {
		t.Fatalf("temporary live frames were retained: %v", err)
	}
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 404 {
		t.Fatalf("stale frame: %d", w.Code)
	}
	path = startTestLive(t, a)
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 204 {
		t.Fatalf("restart did not warm up: %d", w.Code)
	}
	liveRequest(a, "DELETE", path, nil)
}
func TestLiveTimeoutReleasesWorker(t *testing.T) {
	a := testLiveApp(t)
	startTestLive(t, a)
	a.liveMu.Lock()
	s := a.live
	a.liveMu.Unlock()
	s.timer.Reset(time.Millisecond)
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
		t.Fatal("idle worker survived timeout")
	}
	if !a.gpu.TryLock() {
		t.Fatal("GPU reservation survived timeout")
	}
	a.gpu.Unlock()
}

func TestLiveStartupDoesNotRequireBodyAssets(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "present")
	if err := os.WriteFile(model, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	missing := filepath.Join(dir, "missing-body")
	c := config{data: dir, pipeline: model, denoiser: model, vitpose: model, yolox: model, module: model,
		bodyRunner: missing, bodyModule: missing, backbone: missing, branch: missing, mhr: missing,
		backend: "CPU", threads: 8, maxJobs: 1}
	if _, err := load(c); err != nil {
		t.Fatalf("live startup requires Body assets: %v", err)
	}
	c.module = dir // CPU variants are selected from a backend directory.
	if _, err := load(c); err != nil {
		t.Fatalf("backend directory rejected: %v", err)
	}
	c.vitpose = missing
	if _, err := load(c); err == nil {
		t.Fatal("missing live model accepted")
	}
}

func pipelineRequest(a *app, path string, index int, data []byte) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	r := httptest.NewRequest("PUT", path+"/frame", bytes.NewReader(data))
	r.Header.Set("X-GEMX-Frame", strconv.Itoa(index))
	a.route(w, r)
	return w
}

func startPipeline(t *testing.T, a *app) string {
	t.Helper()
	w := liveRequest(a, "POST", "/api/live?pipeline=2", nil)
	if w.Code != 201 {
		t.Fatalf("start pipeline: %d %s", w.Code, w.Body)
	}
	var session struct{ ID string }
	if err := json.Unmarshal(w.Body.Bytes(), &session); err != nil {
		t.Fatal(err)
	}
	return "/api/live/" + session.ID
}

func waitPending(t *testing.T, s *liveSession, index int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		s.queueMu.Lock()
		pending := s.pending[index]
		s.queueMu.Unlock()
		if pending {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("frame was not admitted")
}

func TestLivePipelineOrdersAndBoundsFrames(t *testing.T) {
	a := testLiveApp(t)
	path := startPipeline(t, a)
	frame := livePNG(t, 16)
	second := make(chan *httptest.ResponseRecorder, 1)
	// Simulate HTTP delivery reordering: the next frame arrives first.
	go func() { second <- pipelineRequest(a, path, 2, frame) }()
	waitPending(t, a.live, 2)
	for _, index := range []int{0, 2, 3} {
		if w := pipelineRequest(a, path, index, frame); w.Code != 409 {
			t.Fatalf("index %d admitted: %d", index, w.Code)
		}
	}
	select {
	case <-second:
		t.Fatal("frame 2 ran before frame 1")
	default:
	}
	if w := pipelineRequest(a, path, 1, frame); w.Code != 204 || w.Header().Get("X-GEMX-Sequence") != "1" {
		t.Fatalf("first: %d %s", w.Code, w.Body)
	}
	select {
	case w := <-second:
		if w.Code != 200 || w.Header().Get("X-GEMX-Sequence") != "2" || w.Header().Get("Server-Timing") == "" {
			t.Fatalf("second: %d %s", w.Code, w.Body)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("queued frame stuck")
	}
	if w := pipelineRequest(a, path, 1, frame); w.Code != 409 {
		t.Fatalf("replayed frame accepted: %d", w.Code)
	}
	liveRequest(a, "DELETE", path, nil)
}

func TestLivePipelineCancellationUnblocksQueue(t *testing.T) {
	a := testLiveApp(t)
	path := startPipeline(t, a)
	frame := livePNG(t, 16)
	s := a.live
	second := make(chan *httptest.ResponseRecorder, 1)
	go func() { second <- pipelineRequest(a, path, 2, frame) }()
	waitPending(t, s, 2)
	if w := liveRequest(a, "DELETE", path, nil); w.Code != 204 {
		t.Fatalf("delete: %d", w.Code)
	}
	select {
	case w := <-second:
		if w.Code != 410 {
			t.Fatalf("cancelled frame: %d", w.Code)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cancelled frame stuck")
	}
	if _, err := os.Stat(s.dir); !os.IsNotExist(err) {
		t.Fatalf("live directory retained: %v", err)
	}
}

func TestLiveSourceTime(t *testing.T) {
	a := testLiveApp(t)
	path := startTestLive(t, a)
	frame := livePNG(t, 16)
	send := func(value string) *httptest.ResponseRecorder {
		r := httptest.NewRequest("PUT", path+"/frame", bytes.NewReader(frame))
		r.Header.Set("X-GEMX-Source-Time-Us", value)
		w := httptest.NewRecorder()
		a.route(w, r)
		return w
	}
	for _, value := range []string{"-1", "1.5", "9223372036854775808"} {
		if w := send(value); w.Code != 400 {
			t.Fatalf("invalid time %s: %d", value, w.Code)
		}
	}
	// Preserve int64 timestamps above JavaScript's exact-number range.
	if w := send("9007199254740993"); w.Code != 204 || w.Header().Get("X-GEMX-Source-Time-Us") != "9007199254740993" {
		t.Fatalf("first: %d %v", w.Code, w.Header())
	}
	for _, value := range []string{"9007199254740993", "9007199254740992", ""} {
		if w := send(value); w.Code != 409 {
			t.Fatalf("ordering/mode %s: %d", value, w.Code)
		}
	}
	if w := send("9007199254740994"); w.Code != 200 || w.Header().Get("X-GEMX-Sequence") != "2" {
		t.Fatalf("next: %d %s", w.Code, w.Body)
	}
	command, err := os.ReadFile(filepath.Join(a.live.dir, "last-command"))
	if err != nil || string(command) != "FRAME 2 9007199254740994\n" {
		t.Fatalf("worker source time: %q %v", command, err)
	}
}
