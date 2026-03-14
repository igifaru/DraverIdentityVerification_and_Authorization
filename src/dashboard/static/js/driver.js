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
    const GPS_THRESHOLD_M = 6;    // Trip start trigger
    const REVERIFY_THRESHOLD_M = 5000; // 5km periodic trigger
    const GPS_MIN_ACCURACY = 150; // Relaxed from 50m for better device compatibility

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
    let isStartingCamera = false; // Lock to prevent multiple concurrent start requests
    let prevPosition = null;      // Anchor for "Stationary" vs "Moving" HUD logic
    let lastAuthPosition = null;  // Anchor for the 5km periodic check
    let latestCoords = null;      // Cached latest high-accuracy coordinates
    let gpsWatchId = null;

    /* ── Helpers ──────────────────────────────────────────────────── */
    function setState(s) {
        state = s;
        body.dataset.state = s;
        const labels = {
            waiting_movement: 'WAITING FOR MOVEMENT',
            driving: 'TRIP IN PROGRESS',
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
            const { latitude, longitude, accuracy } = pos.coords;
            
            // Requirement 7: Add console logs
            console.log(`[GPS] Lat: ${latitude.toFixed(6)}, Lon: ${longitude.toFixed(6)} | Accuracy: ${accuracy.toFixed(1)}m`);

            // Requirement 8: Warn about low accuracy but continue (teammate's recommended fix)
            if (accuracy > GPS_MIN_ACCURACY) {
                console.warn(`[GPS] Low accuracy (${accuracy.toFixed(1)}m > ${GPS_MIN_ACCURACY}m) but continuing...`);
                gpsLabel.textContent = `WEAK SIGNAL (±${Math.round(accuracy)}m)`;
            }

            // Cache for general use
            latestCoords = { lat: latitude, lon: longitude };

            // Requirement 3: Store the first detected coordinate as starting position
            if (!prevPosition) {
                prevPosition = { lat: latitude, lon: longitude };
                console.log(`[GPS] Starting position set: ${latitude}, ${longitude}`);
                gpsLabel.textContent = 'SYSTEM ARMED - STATIONARY';
                return;
            }

            // Requirement 4: Continuously compare with starting coordinates using Haversine
            const distance = haversine(prevPosition.lat, prevPosition.lon, latitude, longitude);
            
            // Requirement 7: Log calculated distance
            console.log(`[GPS] Distance from start: ${distance.toFixed(2)}m`);

            // Requirement 5 & 6: Trigger if distance >= 6 meters, with sanity check on accuracy
            if (state === 'waiting_movement') {
                // Professional touch: ensure accuracy is at least within 200m before triggering
                if (distance >= GPS_THRESHOLD_M && accuracy < 200) {
                    console.log(`[GPS] Movement detected: ${distance.toFixed(1)}m. TRIGGERING CAMERA.`);
                    gpsLabel.textContent = `MOVEMENT DETECTED: ${distance.toFixed(1)}m`;
                    gpsStatus.classList.add('moving');
                    
                    // Requirement 10: Stop GPS watcher to avoid repeated triggering
                    if (gpsWatchId !== null) {
                        navigator.geolocation.clearWatch(gpsWatchId);
                        gpsWatchId = null;
                        console.log('[GPS] Watcher stopped after trigger.');
                    }

                    startIdle(); 
                } else {
                    gpsLabel.textContent = `ARMED: ${distance.toFixed(1)}m MOVED`;
                }
            }

        }, (err) => {
            console.error(`[GPS] Error (${err.code}): ${err.message}`);
            gpsLabel.textContent = 'GPS SIGNAL ERROR';
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
        
        // Reset anchors for a fresh trip cycle
        prevPosition = null; 
        lastAuthPosition = null;

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

        // Capture the coordinates where verification succeeded (using cached stream)
        if (latestCoords) {
            lastAuthPosition = { ...latestCoords };
        }

        animateTimer(authTimer, AUTH_HOLD_MS);
        resetTimer = setTimeout(startDrivingMode, AUTH_HOLD_MS); 
    }

    /* ── DRIVING Mode: Trip is in progress, camera OFF ─────── */
    function startDrivingMode() {
        setState('driving');
        verifying = false;
        stopPoll();
        clearStable();
        clearReset();

        // Security/Privacy: Shut down the camera hardware
        fetch('/api/driver/camera/stop', { method: 'POST' }).catch(() => { });
        
        console.log('[driver] Driver Authorized. Entering monitoring mode (5km threshold).');
    }

    /* ── UNAUTHORIZED result ────────────────────────────────────── */
    function showUnauthorized(data) {
        setState('unauthorized');
        if (data.status_message) {
            deniedReason.textContent = data.status_message;
        } else {
            deniedReason.textContent = 'Unauthorized access attempt';
        }
        animateTimer(deniedTimer, DENY_HOLD_MS);
        resetTimer = setTimeout(startMovementWait, DENY_HOLD_MS); // Go back to waiting for next movement
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