/**
 * driver.js — Driver Verification Terminal
 *
 * State machine:
 *
 *   IDLE  ──(face detected)──► DETECTING  ──(stable 2s)──► PROCESSING
 *                 ▲                 |                            |
 *                 │          (face lost)                    (result)
 *                 │            returns                      /       \
 *                 └──────────── IDLE          AUTHORIZED    UNAUTHORIZED
 *                                               (3s)  ───────► IDLE
 *                                            UNAUTHORIZED
 *                                               (5s)  ───────► IDLE
 *
 * APIs:
 *   GET  /api/driver/detect   → {face_present, confidence}
 *   POST /api/driver/verify   → {state, driver_name, similarity, event_id}
 */

(function () {
    'use strict';

    /* ── Timing & GPS constants ───────────────────────────────────── */
    const DETECT_POLL_MS = 800;    // idle poll interval
    const STABLE_MS = 2000;   // face must be present this long before verify
    const AUTH_HOLD_MS = 3000;   // green result shown for
    const DENY_HOLD_MS = 5000;   // red result shown for
    const CLOCK_MS = 1000;   // clock refresh
    const GPS_THRESHOLD_M = 6;    // distance in meters to trigger verification
    const GPS_MIN_ACCURACY = 50;  // maximum acceptable accuracy in meters

    /* ── DOM refs ─────────────────────────────────────────────────── */
    const body = document.body;
    const countBar = document.getElementById('countdownBar');
    const authTimer = document.getElementById('authTimer');
    const deniedTimer = document.getElementById('deniedTimer');
    const driverName = document.getElementById('driverName');
    const authMeta = document.getElementById('authMeta');
    const stateLed = document.getElementById('stateLed');
    const stateWord = document.getElementById('stateWord');
    const clockEl = document.getElementById('clockEl');
    const gpsStatus = document.getElementById('gpsStatus');
    const gpsLabel = document.getElementById('gpsLabel');

    /* ── State ────────────────────────────────────────────────────── */
    let state = 'idle';
    let pollTimer = null;
    let stableTimer = null;
    let resetTimer = null;
    let checkStableTimer = null;
    let verifying = false;
    let isStartingCamera = false; // Lock to prevent multiple concurrent start requests
    let prevPosition = null;      // Last recorded GPS position {lat, lon}
    let gpsWatchId = null;

    /* ── Helpers ──────────────────────────────────────────────────── */
    function setState(s) {
        state = s;
        body.dataset.state = s;
        const labels = {
            waiting_movement: 'WAITING FOR MOVEMENT',
            idle: 'SCANNING',
            detecting: 'FACE DETECTED',
            processing: 'VERIFYING',
            authorized: 'AUTHORIZED',
            unauthorized: 'ACCESS DENIED',
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
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    }

    /* ── GPS Management ─────────────────────────────────────────── */
    function initGPS() {
        if (!navigator.geolocation) {
            gpsLabel.textContent = 'GPS NOT SUPPORTED';
            gpsStatus.style.color = 'var(--red)';
            return;
        }

        const options = {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 0
        };

        gpsWatchId = navigator.geolocation.watchPosition((pos) => {
            const { latitude, longitude, accuracy } = pos.coords;
            console.log(`[GPS] Lat: ${latitude}, Lon: ${longitude}, Acc: ${accuracy}m`);

            if (accuracy > GPS_MIN_ACCURACY) {
                gpsLabel.textContent = `GPS ACCURACY LOW (${Math.round(accuracy)}m)`;
                gpsStatus.classList.remove('moving');
                return;
            }

            if (!prevPosition) {
                prevPosition = { lat: latitude, lon: longitude };
                gpsLabel.textContent = 'VEHICLE STATIONARY';
                gpsStatus.classList.remove('moving');
                return;
            }

            const distance = haversine(prevPosition.lat, prevPosition.lon, latitude, longitude);
            console.log(`[GPS] Distance: ${distance.toFixed(2)}m`);

            if (distance >= GPS_THRESHOLD_M) {
                console.log(`[GPS] Movement detected: ${distance.toFixed(2)}m. Triggering verification...`);
                gpsLabel.textContent = `MOVING: ${distance.toFixed(1)}m`;
                gpsStatus.classList.add('moving');

                // Trigger face verification if we're not already in a process
                if (state === 'waiting_movement' || state === 'idle') {
                    startIdle(); // This will start the face detection poll
                }

                // Update previous position only after hitting the threshold as per diagram logic 
                // "The system should run continuously, updating the previous coordinate after each calculation"
                prevPosition = { lat: latitude, lon: longitude };

                // Reset to "STATIONARY" label after a short delay if movement stops being detected
                setTimeout(() => {
                    if (state === 'idle' || state === 'waiting_movement') {
                        gpsLabel.textContent = 'VEHICLE STATIONARY';
                        gpsStatus.classList.remove('moving');
                    }
                }, 5000);
            } else {
                gpsLabel.textContent = 'VEHICLE STATIONARY';
                gpsStatus.classList.remove('moving');
            }

        }, (err) => {
            console.warn('[GPS] error:', err.message);
            gpsLabel.textContent = 'GPS SIGNAL LOST';
            gpsStatus.style.color = 'var(--red)';
        }, options);
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

    /* ── Movement Wait: Initial state ──────────────────────────── */
    function startMovementWait() {
        if (state === 'waiting_movement') return;
        setState('waiting_movement');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        // Stop camera if it was on
        fetch('/api/camera/stop', { method: 'POST' }).catch(() => { });

        console.log('[driver] System armed: Waiting for vehicle movement...');
    }

    /* ── IDLE: poll for a face ──────────────────────────────────── */
    async function startIdle() {
        if (state === 'idle') return;
        setState('idle');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        async function poll() {
            if (state !== 'idle' || pollTimer === null) return;

            // STRICT CHECK: Only poll if the tab is visible AND the window has focus.
            const isVisible = document.visibilityState === 'visible';
            const isFocused = document.hasFocus();

            if (!isVisible || !isFocused) {
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
                        console.log('[driver] Activating camera after movement...');
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
                console.warn('[driver] detect error:', e.message);
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
        if (document.visibilityState !== 'visible' || !document.hasFocus()) return;
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
                startIdle();
            }
        } catch (e) {
            startIdle();
        }
    }

    /* ── AUTHORIZED result ──────────────────────────────────────── */
    function showAuthorized(data) {
        setState('authorized');
        driverName.textContent = data.driver_name ? `Welcome, ${data.driver_name}` : 'Driver Identified';
        authMeta.textContent = data.similarity ? `Match confidence: ${(data.similarity * 100).toFixed(1)}%` : '';

        animateTimer(authTimer, AUTH_HOLD_MS);
        resetTimer = setTimeout(startMovementWait, AUTH_HOLD_MS); // Go back to waiting for next movement
    }

    /* ── UNAUTHORIZED result ────────────────────────────────────── */
    function showUnauthorized(data) {
        setState('unauthorized');
        animateTimer(deniedTimer, DENY_HOLD_MS);
        resetTimer = setTimeout(startMovementWait, DENY_HOLD_MS); // Go back to waiting for next movement
    }

    /* ── Boot ───────────────────────────────────────────────────── */
    function checkAndStart() {
        if (document.visibilityState === 'visible' && document.hasFocus()) {
            if (gpsWatchId === null) {
                initGPS();
            }
            if (state === 'idle' || state === 'waiting_movement' || body.dataset.state === 'idle') {
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