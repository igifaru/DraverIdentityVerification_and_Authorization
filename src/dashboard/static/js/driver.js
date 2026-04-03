/**
 * driver.js — Driver Verification Terminal
 *
 * GPS-driven state machine:
 *
 *   STATE 1: WAITING_FOR_MOVEMENT
 *       GPS watches position vs saved anchor.
 *       distance >= 6m  ──────────────────────► STATE 2 (start camera + scan)
 *
 *   STATE 2: CAPTURING  (idle → detecting → processing → result)
 *       Camera ON. Face detected, verified.
 *       On result (auth or unauth) ───────────► STOP CAMERA → save location → STATE 3
 *
 *   STATE 3: COOLDOWN
 *       Camera OFF. GPS re-anchors to capture location.
 *       distance >= 10m from capture point ──► RESET → STATE 1
 *
 * APIs:
 *   GET  /api/driver/detect        → {face_present, confidence}
 *   POST /api/driver/verify        → {state, driver_name, similarity, event_id}
 *   POST /api/location/update      → {state, lat, lon, distance_m}  (telemetry log)
 */

(function () {
    'use strict';

    /* ── Timing & GPS constants ───────────────────────────────────── */
    const DETECT_POLL_MS        = 800;   // idle poll interval (ms)
    const STABLE_MS             = 2000;  // face must stay present before verify (ms)
    const AUTH_HOLD_MS          = 3000;  // green result display duration (ms)
    const DENY_HOLD_MS          = 5000;  // red result display duration (ms)
    const CLOCK_MS              = 1000;  // clock refresh (ms)
    const GPS_START_THRESHOLD_M = 6;     // STATE 1→2: movement trigger distance (metres)
    const GPS_COOLDOWN_RESET_M  = 10;    // STATE 3→1: cooldown exit distance (metres)
    const GPS_MIN_ACCURACY      = 150;   // max accepted accuracy radius (metres)
    const GPS_MIN_SPEED_MS      = 0.5;   // 0.5 m/s ≈ 1.8 km/h — min speed to confirm movement

    /* ── DOM refs ─────────────────────────────────────────────────── */
    const body = document.body;
    const countBar = document.getElementById('countdownBar');
    const authTimer = document.getElementById('authTimer');
    const deniedTimer = document.getElementById('deniedTimer');
    const deniedReason = document.getElementById('deniedReason');
    const driverName = document.getElementById('driverName');
    const authMeta = document.getElementById('authMeta');
    const stateLed = document.getElementById('stateLed');
    const stateWord = document.getElementById('stateWord');
    const clockEl = document.getElementById('clockEl');
    const gpsStatus = document.getElementById('gpsStatus');
    const gpsLabel = document.getElementById('gpsLabel');

    /* ── State ────────────────────────────────────────────────────── */
    let state = 'boot';
    let pollTimer = null;
    let stableTimer = null;
    let resetTimer = null;
    let checkStableTimer = null;
    let verifying = false;
    let isStartingCamera = false;    // Lock to prevent multiple concurrent start requests
    let prevPosition = null;         // STATE 1 anchor: position when WAITING_FOR_MOVEMENT began
    let capturePosition = null;      // STATE 3 anchor: position where capture happened
    let latestCoords = null;         // Cached latest high-accuracy coordinates
    let gpsWatchId = null;
    let movementConfirmCount = 0;    // Consecutive readings above threshold (noise guard - STATE 1)
    let cooldownConfirmCount = 0;     // Consecutive readings above 10m threshold (noise guard - STATE 3)

    /* ── Helpers ──────────────────────────────────────────────────── */
    function setState(s) {
        state = s;
        body.dataset.state = s;
        const labels = {
            waiting_movement: 'WAITING FOR MOVEMENT',
            idle:             'SCANNING',
            detecting:        'FACE DETECTED',
            processing:       'VERIFYING',
            authorized:       'AUTHORIZED',
            unauthorized:     'ACCESS DENIED',
            cooldown:         'COOLDOWN — MOVE 10m TO RESET',
        };
        stateWord.textContent = labels[s] || s.toUpperCase();
    }

    function haversine(lat1, lon1, lat2, lon2) {
        const R = 6371000; // Earth radius in meters
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
        return R * c;
    }

    /* ── GPS Management ─────────────────────────────────────────── */
    function initGPS() {
        if (!navigator.geolocation) {
            console.error('[GPS] Geolocation is not supported by this browser.');
            gpsLabel.textContent = 'GPS NOT SUPPORTED';
            gpsStatus.style.color = 'var(--red)';
            return;
        }

        const options = {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        };

        console.log('[GPS] Starting high-accuracy watcher...');

        gpsWatchId = navigator.geolocation.watchPosition((pos) => {
            const { latitude, longitude, accuracy, speed } = pos.coords;

            console.log(`[GPS] Lat: ${latitude.toFixed(6)}, Lon: ${longitude.toFixed(6)} | Accuracy: ${accuracy.toFixed(1)}m | Speed: ${speed !== null ? speed.toFixed(2) + 'm/s' : 'n/a'}`);

            if (accuracy > GPS_MIN_ACCURACY) {
                console.warn(`[GPS] Low accuracy (${accuracy.toFixed(1)}m > ${GPS_MIN_ACCURACY}m) but continuing...`);
                gpsLabel.textContent = `WEAK SIGNAL (±${Math.round(accuracy)}m)`;
            }

            // Always cache latest coords
            latestCoords = { lat: latitude, lon: longitude };

            // ─────────────────────────────────────────────────────────────
            // STATE 1: WAITING_FOR_MOVEMENT
            //   Anchor set on first fix. Trigger camera when >= 6m moved.
            // ─────────────────────────────────────────────────────────────
            if (state === 'waiting_movement') {

                if (!prevPosition) {
                    prevPosition = { lat: latitude, lon: longitude };
                    console.log(`[GPS] STATE1 anchor set: ${latitude.toFixed(6)}, ${longitude.toFixed(6)}`);
                    gpsLabel.textContent = 'ARMED — WAITING FOR MOVEMENT';
                    postLocation('waiting_movement', latitude, longitude, 0);
                    return;
                }

                const distance = haversine(prevPosition.lat, prevPosition.lon, latitude, longitude);
                console.log(`[GPS] STATE1 distance from anchor: ${distance.toFixed(2)}m`);
                gpsLabel.textContent = `ARMED: ${distance.toFixed(1)}m MOVED`;

                if (distance >= GPS_START_THRESHOLD_M && accuracy < GPS_MIN_ACCURACY) {
                    const speedAvailable = speed !== null && speed !== undefined;

                    if (speedAvailable && speed >= GPS_MIN_SPEED_MS) {
                        // Speed sensor confirms real movement — trigger immediately
                        movementConfirmCount = 0;
                        console.log(`[GPS] STATE1→CAPTURE (speed confirmed): ${distance.toFixed(1)}m @ ${speed.toFixed(2)}m/s`);
                        gpsLabel.textContent = `MOVEMENT DETECTED: ${distance.toFixed(1)}m`;
                        gpsStatus.classList.add('moving');
                        postLocation('movement_triggered', latitude, longitude, distance);
                        startIdle();

                    } else if (!speedAvailable) {
                        // No speed sensor — require 3 consecutive readings above threshold
                        movementConfirmCount++;
                        console.log(`[GPS] STATE1 movement reading ${movementConfirmCount}/3: ${distance.toFixed(1)}m`);
                        gpsLabel.textContent = `CONFIRMING MOVEMENT (${movementConfirmCount}/3): ${distance.toFixed(1)}m`;
                        if (movementConfirmCount >= 3) {
                            movementConfirmCount = 0;
                            console.log(`[GPS] STATE1→CAPTURE (3× confirmed): ${distance.toFixed(1)}m`);
                            gpsStatus.classList.add('moving');
                            postLocation('movement_triggered', latitude, longitude, distance);
                            startIdle();
                        }
                    }
                    // Speed available but below threshold → GPS noise, ignore

                } else {
                    movementConfirmCount = 0;
                }
            }

            // ─────────────────────────────────────────────────────────────
            // STATE 3: COOLDOWN
            //   Anchor = capture location. Reset to STATE 1 when >= 10m moved.
            // ─────────────────────────────────────────────────────────────
            else if (state === 'cooldown' && capturePosition) {
                const distFromCapture = haversine(capturePosition.lat, capturePosition.lon, latitude, longitude);
                console.log(`[GPS] STATE3 distance from capture: ${distFromCapture.toFixed(2)}m`);
                gpsLabel.textContent = `COOLDOWN: ${distFromCapture.toFixed(1)}m / ${GPS_COOLDOWN_RESET_M}m`;

                if (distFromCapture >= GPS_COOLDOWN_RESET_M) {
                    const speedAvailable = speed !== null && speed !== undefined;

                    if (speedAvailable && speed >= GPS_MIN_SPEED_MS) {
                        // Speed confirms real movement - reset immediately
                        cooldownConfirmCount = 0;
                        console.log(`[GPS] STATE3->STATE1 (speed confirmed): ${distFromCapture.toFixed(1)}m`);
                        postLocation('cooldown_reset', latitude, longitude, distFromCapture);
                        startMovementWait();

                    } else if (!speedAvailable) {
                        // No speed sensor - require 3 consecutive readings above 10m
                        cooldownConfirmCount++;
                        console.log(`[GPS] STATE3 reset reading ${cooldownConfirmCount}/3: ${distFromCapture.toFixed(1)}m`);
                        gpsLabel.textContent = `COOLDOWN: confirming reset (${cooldownConfirmCount}/3)`;
                        if (cooldownConfirmCount >= 3) {
                            cooldownConfirmCount = 0;
                            console.log(`[GPS] STATE3->STATE1 (3x confirmed): ${distFromCapture.toFixed(1)}m`);
                            postLocation('cooldown_reset', latitude, longitude, distFromCapture);
                            startMovementWait();
                        }
                    }
                    // Speed available but below threshold - GPS drift noise, ignore

                } else {
                    // Distance dropped back below 10m - reset confirmation streak
                    cooldownConfirmCount = 0;
                }
            }

        }, (err) => {
            console.error(`[GPS] Error (${err.code}): ${err.message}`);
            gpsLabel.textContent = 'GPS SIGNAL ERROR';
            gpsStatus.style.color = 'var(--red)';
        }, options);
    }

    /* ── Telemetry: POST location update to backend ─────────────── */
    function postLocation(eventState, lat, lon, distance_m) {
        fetch('/api/location/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ state: eventState, lat, lon, distance_m: Math.round(distance_m * 10) / 10 })
        }).catch(() => { /* fire-and-forget */ });
    }


    function clearReset() {
        if (resetTimer) { clearTimeout(resetTimer); resetTimer = null; }
    }

    function clearStable() {
        if (stableTimer) { clearTimeout(stableTimer); stableTimer = null; }
        if (checkStableTimer) { clearInterval(checkStableTimer); checkStableTimer = null; }
    }

    function stopPoll() {
        if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    }

    /* ── Clock ────────────────────────────────────────────────────── */
    (function tickClock() {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
        setTimeout(tickClock, CLOCK_MS);
    }());

    /* ── Animate a timer bar draining from 100% → 0% ── */
    function animateTimer(el, durationMs) {
        el.style.transition = 'none';
        el.style.transform = 'scaleX(1)';
        void el.offsetWidth; // force reflow
        el.style.transition = `transform ${durationMs}ms linear`;
        el.style.transform = 'scaleX(0)';
    }

    /* ── Animate the countdown bar filling 0% → 100% ── */
    function animateCountdown(durationMs) {
        countBar.style.transition = 'none';
        countBar.style.width = '0%';
        void countBar.offsetWidth;
        countBar.style.transition = `width ${durationMs}ms linear`;
        countBar.style.width = '100%';
    }

    /* ── STATE 1: WAITING_FOR_MOVEMENT ─────────────────────────── */
    function startMovementWait() {
        if (state === 'waiting_movement') return;
        setState('waiting_movement');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        // Fresh trip cycle — reset all anchors and counters
        prevPosition = null;
        capturePosition = null;
        movementConfirmCount = 0;
        cooldownConfirmCount = 0;
        gpsStatus.classList.remove('moving');

        // Restore video feed src if it was blanked during cooldown
        const videoFeed = document.getElementById('videoFeed');
        if (videoFeed && !videoFeed.src && videoFeed._originalSrc) {
            videoFeed.src = videoFeed._originalSrc;
        }

        if (gpsWatchId === null) {
            initGPS();
        }

        fetch('/api/driver/camera/stop', { method: 'POST' }).catch(() => { });
    }

    /* ── IDLE: poll for a face ──────────────────────────────────── */
    async function startIdle() {
        if (state === 'idle') return;
        setState('idle');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        // Ensure moving effect is off when we arrive at scanning UI
        gpsStatus.classList.remove('moving');

        async function poll() {
            if (state !== 'idle' || pollTimer === null) return;

            // KIOSK MODE: Only check visibility, ignore focus as drivers don't click terminals.
            if (document.visibilityState !== 'visible') {
                pollTimer = setTimeout(poll, DETECT_POLL_MS);
                return;
            }

            try {
                const res = await fetch('/api/driver/detect');
                const data = await res.json();

                // If the backend says camera is OFF, wake it up!
                if (data.camera_off) {
                    if (!isStartingCamera) {
                        isStartingCamera = true;
                        try {
                            const startRes = await fetch('/api/driver/camera/start', { method: 'POST' });
                            const startData = await startRes.json();
                        } finally {
                            isStartingCamera = false;
                        }
                    }
                    pollTimer = setTimeout(poll, 1500);
                    return;
                }

                if (data.face_present) {
                    onFaceDetected();
                    return;
                }
            } catch (e) {
            }
            pollTimer = setTimeout(poll, DETECT_POLL_MS);
        }

        pollTimer = setTimeout(poll, DETECT_POLL_MS);
    }

    /* ── DETECTING: stable 2-second countdown ──────────────────── */
    function onFaceDetected() {
        if (state !== 'idle') return;
        setState('detecting');
        stopPoll();
        animateCountdown(STABLE_MS);

        let lostCount = 0;
        checkStableTimer = setInterval(async () => {
            try {
                const res = await fetch('/api/driver/detect');
                const data = await res.json();
                if (!data.face_present) {
                    lostCount++;
                    if (lostCount >= 2) {
                        clearStable();
                        startIdle();
                        return;
                    }
                } else {
                    lostCount = 0;
                }
            } catch (e) { }
        }, 400);

        stableTimer = setTimeout(() => {
            clearStable();
            startVerification();
        }, STABLE_MS);
    }

    /* ── PROCESSING: call /api/driver/verify ───────────────────── */
    async function startVerification() {
        if (document.visibilityState !== 'visible') return;
        if (verifying) return;
        verifying = true;
        setState('processing');

        try {
            const res = await fetch('/api/driver/verify', { method: 'POST' });
            const data = await res.json();

            if (data.state === 'authorized') {
                showAuthorized(data);
            } else if (data.state === 'unauthorized') {
                showUnauthorized(data);
            } else {
                // no_face or unexpected state — one attempt was made; shut camera off
                fetch('/api/driver/camera/stop', { method: 'POST' }).catch(() => { });
                startMovementWait();
            }
        } catch (e) {
            // Network failure — shut camera off rather than leaving it running
            fetch('/api/driver/camera/stop', { method: 'POST' }).catch(() => { });
            startMovementWait();
        }
    }

    /* ── AUTHORIZED result ──────────────────────────────────────── */
    function showAuthorized(data) {
        setState('authorized');
        driverName.textContent = data.driver_name ? `Welcome, ${data.driver_name}` : 'Driver Identified';
        authMeta.textContent = data.similarity ? `Match confidence: ${(data.similarity * 100).toFixed(1)}%` : '';

        // Save capture location as the cooldown anchor
        if (latestCoords) {
            capturePosition = { ...latestCoords };
            postLocation('capture_authorized', capturePosition.lat, capturePosition.lon, 0);
        }

        animateTimer(authTimer, AUTH_HOLD_MS);
        resetTimer = setTimeout(startCooldown, AUTH_HOLD_MS);
    }

    /* ── STATE 3: COOLDOWN ──────────────────────────────────────── */
    function startCooldown() {
        setState('cooldown');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        // Stop the camera backend stream
        fetch('/api/driver/camera/stop', { method: 'POST' }).catch(() => { });

        // Blank the video feed element so no stale frame shows through
        // (CSS sets display:none but clearing src ensures no cached frame leaks)
        const videoFeed = document.getElementById('videoFeed');
        if (videoFeed) {
            videoFeed._originalSrc = videoFeed.src;
            videoFeed.src = '';
        }

        cooldownConfirmCount = 0;  // Fresh cooldown cycle
        console.log('[driver] STATE3: COOLDOWN. Screen OFF. Camera OFF. Move 10m to reset.');
        gpsLabel.textContent = `COOLDOWN: 0m / ${GPS_COOLDOWN_RESET_M}m`;
    }

    /* ── DRIVING Mode (kept for compatibility, redirects to cooldown) */
    function startDrivingMode() {
        startCooldown();
    }

    /* ── UNAUTHORIZED result ────────────────────────────────────── */
    function showUnauthorized(data) {
        setState('unauthorized');
        if (data.status_message) {
            deniedReason.textContent = data.status_message;
        } else {
            deniedReason.textContent = 'Unauthorized access attempt';
        }

        // Save capture location even on denial — cooldown still applies
        if (latestCoords) {
            capturePosition = { ...latestCoords };
            postLocation('capture_unauthorized', capturePosition.lat, capturePosition.lon, 0);
        }

        animateTimer(deniedTimer, DENY_HOLD_MS);
        resetTimer = setTimeout(startCooldown, DENY_HOLD_MS);
    }

    /* ── Boot ───────────────────────────────────────────────────── */
    function checkAndStart() {
        if (document.visibilityState === 'visible') {
            if (gpsWatchId === null) {
                initGPS();
            }
            // Only force arm if we are in the initial boot state
            if (state === 'boot') {
                startMovementWait();
            }
        } else {
            stopPoll();
            clearStable();
        }
    }

    // Run once on load
    checkAndStart();

    document.addEventListener('visibilitychange', checkAndStart);
    window.addEventListener('focus', checkAndStart);
    window.addEventListener('blur', checkAndStart);

}());